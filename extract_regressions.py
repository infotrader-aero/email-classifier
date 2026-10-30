#!/usr/bin/env python3
"""
Download regression emails for checking that classifier changes don't break
request_for_quote and vendor_quote_response classification.

Samples emails that created RFQs or incoming quotes, as they normally arrive,
plus the ones prod mislabeled (the hardest cases). Only emails received on or
after --since are used, so the current model wasn't trained on them.

New emails go to regressions/pending_review/ (<id>.eml raw, <id>.txt readable)
and regressions/manifest.json, without an expected label. Once an email is
hand-verified, set its `expected` label and `set` in the manifest and move its
files to regressions/<expected>/. Run check_regressions.py afterwards.
Each run adds to the existing set, skipping emails already in it.
"""

import json
import argparse
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from app.database import get_connection, fetch_inbound_emails_by_ids
from app.storage import get_s3_client, download_emails_batch
from app.email_parser import parse_email, extract_text_for_classification
from app.regressions import REGRESSIONS_DIR, MANIFEST_PATH, PENDING_REVIEW

HAS_RFQ = "EXISTS (SELECT 1 FROM request_for_quotes rfq WHERE rfq.inbound_email_id = amie.id)"
HAS_IQ = "EXISTS (SELECT 1 FROM incoming_quote_inbound_emails iqie WHERE iqie.inbound_email_id = amie.id)"
NO_QUOTE_SQL = r"(\mno quote\M|\mno bid\M|\mno stock\M|\mnq\M|unable to quote|regret)"

# (source, where clause, params, number of emails)
SAMPLES = [
    ('created rfq, prod request_for_quote', f"{HAS_RFQ} AND amie.category_label = 'request_for_quote'", [], 150),
    ('created rfq, prod listing_service', f"{HAS_RFQ} AND amie.category_label = 'listing_service'", [], 100),
    ('created rfq, prod other label',
     f"{HAS_RFQ} AND amie.category_label NOT IN ('request_for_quote', 'listing_service')", [], 150),
    ('created rfq, OnAv', f"{HAS_RFQ} AND t.name = 'OnAv'", [], 100),
    ('incoming quote, Aero-Zone', f"{HAS_IQ} AND t.name = 'Aero-Zone'", [], 150),
    ('incoming quote, OnAv', f"{HAS_IQ} AND t.name = 'OnAv'", [], 150),
    ('incoming quote, prod other label',
     f"{HAS_IQ} AND amie.category_label NOT IN ('vendor_quote_response', 'vendor_promotion')", [], 100),
    ('no quote reply',
     "EXISTS (SELECT 1 FROM email_training_samples ets WHERE ets.inbound_email_id = amie.id"
     " AND (ets.subject || ' ' || ets.body_text) ~* %s)", [NO_QUOTE_SQL], 100),
]


def sample_ids(conn, since, where, params, limit, exclude_ids):
    """Deterministically sample inbound email ids received on or after a date."""
    query = f"""
        SELECT amie.id
        FROM action_mailbox_inbound_emails amie
        LEFT JOIN tenants t ON t.id = amie.tenant_id
        WHERE amie.created_at >= %s AND {where}
        ORDER BY md5(amie.id::text)
        LIMIT %s
    """
    with conn.cursor() as cursor:
        cursor.execute(query, [since, *params, limit + len(exclude_ids)])
        ids = [row['id'] for row in cursor.fetchall() if row['id'] not in exclude_ids]
    return ids[:limit]


def select_emails(conn, since, exclude_ids, scale):
    """Select email ids for each sample. Returns {email_id: source}."""
    selected = {}
    for source, where, params, limit in SAMPLES:
        for email_id in sample_ids(conn, since, where, params, round(limit * scale), exclude_ids | set(selected)):
            selected[email_id] = source
    return selected


def write_readable(path, email, text):
    """Write a readable copy of the email, in the same layout as extract_examples.py."""
    parsed = email['parsed']
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ID: {email['id']}\n")
        f.write(f"Source: {email['source']}\n")
        f.write(f"Tenant: {email['tenant']}\n")
        f.write(f"Production Category: {email['production_category']}\n")
        f.write(f"Date: {email['created_at']}\n")
        f.write("\n" + "-" * 40 + "\n\n")
        f.write(f"From: {parsed.get('from', '')}\n")
        f.write(f"To: {parsed.get('to', '')}\n")
        f.write(f"Subject: {parsed.get('subject', '')}\n")
        if parsed.get('attachment_names'):
            f.write(f"Attachments: {', '.join(parsed['attachment_names'])}\n")
        f.write("\n" + "-" * 40 + "\n")
        f.write("BODY:\n" + "-" * 40 + "\n\n")
        f.write(text)
        f.write("\n\n" + "=" * 80 + "\n")


def extract_regressions(since, scale):
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []

    print("Connecting to database...")
    conn = get_connection()
    selected = select_emails(conn, since, {email['id'] for email in manifest}, scale)
    records = fetch_inbound_emails_by_ids(conn, selected)

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT amie.id, t.name
            FROM action_mailbox_inbound_emails amie
            LEFT JOIN tenants t ON t.id = amie.tenant_id
            WHERE amie.id = ANY(%s)
        """, (list(selected),))
        tenants = {row['id']: row['name'] for row in cursor.fetchall()}
    conn.close()

    print(f"Downloading {len(records)} emails...")
    set_dir = REGRESSIONS_DIR / PENDING_REVIEW
    set_dir.mkdir(parents=True, exist_ok=True)
    added = 0

    for record, raw_content in download_emails_batch(get_s3_client(), records):
        email_id = record['id']
        try:
            parsed = parse_email(raw_content)
        except Exception as e:
            print(f"Error parsing email {email_id}: {e}")
            continue
        text = extract_text_for_classification(parsed)
        if not text:
            continue

        email = {
            'id': email_id,
            'set': PENDING_REVIEW,
            'source': selected[email_id],
            'expected': None,
            'tenant': tenants.get(email_id),
            'production_category': record['category_label'],
            'created_at': str(record['created_at']),
            'subject': parsed.get('subject', ''),
        }
        (set_dir / f"{email_id}.eml").write_bytes(raw_content)
        write_readable(set_dir / f"{email_id}.txt", {**email, 'parsed': parsed}, text)
        manifest.append(email)
        added += 1

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAdded {added} emails to {set_dir}/ for review ({len(manifest)} in {MANIFEST_PATH})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download RFQ and vendor quote response regression emails')
    parser.add_argument('--since', type=str, default='2026-04-14',
                        help='Only use emails received on or after this date (YYYY-MM-DD, default: 2026-04-14)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Multiply every sample size by this (default: 1.0, about 1000 emails)')
    args = parser.parse_args()

    extract_regressions(date.fromisoformat(args.since), args.scale)
