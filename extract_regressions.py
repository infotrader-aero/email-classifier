#!/usr/bin/env python3
"""
Download a regression set of emails for checking classification.

Repair service quotes look a lot like vendor quote responses (IQs), so besides
repair quotes this downloads the emails most likely to be confused with them.
Only emails received on or after --since (the training cutoff) are used, so
none of the negatives were seen in training.

Writes regressions/<set>/<id>.eml (raw), <id>.txt (readable) and
regressions/manifest.json. Run check_regressions.py afterwards.
"""

import sys
import json
import argparse
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from app.database import get_connection, fetch_inbound_emails_by_ids, fetch_repair_service_quote_candidate_ids
from app.storage import get_s3_client, download_email_content
from app.email_parser import parse_email, extract_text_for_classification
from app.categories import is_repair_service_quote, REPAIR_SERVICE_QUOTE
from app.regressions import REGRESSIONS_DIR, MANIFEST_PATH, is_regression_holdout

# Emails from INF-5509: ILS notice for D5757062000200, its forwards, and the shop's direct quote
TICKET_EXAMPLE_IDS = [317937, 326082, 326088, 317338]

# Repair-ish wording that makes an email easy to confuse with a repair quote
REPAIR_WORDING_SQL = r"(repair|overhaul|bench|\mOH\M|\mRP\M|service:|\mTAT\M|eval)"

NEEDS_REVIEW = 'needs_review'

# Emails awaiting a hand-verified label
PENDING_REVIEW = 'pending_review'

# Core regression samples: the categories that matter most, sampled as they
# normally arrive. Emails that created RFQs or incoming quotes are the closest
# thing to ground truth, and the ones prod mislabeled are the hardest cases.
HAS_RFQ = "EXISTS (SELECT 1 FROM request_for_quotes rfq WHERE rfq.inbound_email_id = amie.id)"
HAS_IQ = "EXISTS (SELECT 1 FROM incoming_quote_inbound_emails iqie WHERE iqie.inbound_email_id = amie.id)"
NO_QUOTE_SQL = r"(\mno quote\M|\mno bid\M|\mno stock\M|\mnq\M|unable to quote|regret)"
CORE_SAMPLES = [
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


def sample_inbound_ids(conn, since, where, params, limit, exclude_ids):
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


def select_core_emails(conn, since, exclude_ids):
    """Select core regression samples, pending review. Returns {email_id: (set, source)}."""
    selected = {}
    for source, where, params, limit in CORE_SAMPLES:
        for email_id in sample_inbound_ids(conn, since, where, params, limit, exclude_ids | set(selected)):
            selected[email_id] = (PENDING_REVIEW, source)
    return selected


def sample_ids(conn, where, params, limit, exclude_ids):
    """Deterministically sample email ids from email_training_samples."""
    query = f"""
        SELECT DISTINCT ON (ets.inbound_email_id) ets.inbound_email_id AS id, md5(ets.inbound_email_id::text) AS h
        FROM email_training_samples ets
        JOIN action_mailbox_inbound_emails amie ON amie.id = ets.inbound_email_id
        WHERE amie.created_at >= %s AND {where}
    """
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT id FROM ({query}) s ORDER BY h", params)
        ids = [row['id'] for row in cursor.fetchall() if row['id'] not in exclude_ids]
    return ids[:limit]


def select_emails(conn, since, num):
    """Select email ids for each regression set. Returns {email_id: (set, reason)}."""
    selected = {}

    candidate_ids = fetch_repair_service_quote_candidate_ids(conn, since=since)
    for email_id in candidate_ids:
        if is_regression_holdout(email_id):
            selected[email_id] = (REPAIR_SERVICE_QUOTE, 'holdout candidate')
    for email_id in TICKET_EXAMPLE_IDS:
        selected[email_id] = (REPAIR_SERVICE_QUOTE, 'INF-5509 example')

    # Training candidates stay eligible as negatives: one the rule matches shows up as a flip
    exclude = set(TICKET_EXAMPLE_IDS)
    has_iq = "EXISTS (SELECT 1 FROM incoming_quote_inbound_emails iqie WHERE iqie.inbound_email_id = ets.inbound_email_id)"
    negatives = [
        ('vendor_quote_response', 'incoming quote', has_iq, [since]),
        ('vendor_quote_response', 'incoming quote, repair wording',
         f"{has_iq} AND (ets.subject || ' ' || ets.body_text) ~* %s", [since, REPAIR_WORDING_SQL]),
        ('request_for_quote', 'rfq, repair wording',
         "ets.corrected_category = 'request_for_quote' AND (ets.subject || ' ' || ets.body_text) ~* %s",
         [since, REPAIR_WORDING_SQL]),
        # Our outbound form only: shop replies to it are usually repair quotes
        ('request_for_quote', 'ils service rfq form',
         r"ets.body_text ~* 'service:\s*(rp|oh|bc|ts)\s*,' AND ets.subject !~* '^\s*(re|fw|fwd):'", [since]),
        ('vendor_response', 'vendor response, repair wording',
         "ets.corrected_category = 'vendor_response' AND (ets.subject || ' ' || ets.body_text) ~* %s",
         [since, REPAIR_WORDING_SQL]),
    ]
    for set_name, reason, where, params in negatives:
        for email_id in sample_ids(conn, where, params, num, exclude | set(selected)):
            selected[email_id] = (set_name, reason)

    return selected


def write_readable(path, email, text):
    """Write a readable copy of the email, in the same layout as extract_examples.py."""
    parsed = email['parsed']
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ID: {email['id']}\n")
        f.write(f"Regression Set: {email['set']} ({email['source']})\n")
        f.write(f"Grok Category: {email['grok_category']}\n")
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


def extract_regressions(since, num, core=False):
    print("Connecting to database...")
    conn = get_connection()

    # Core samples are appended to an existing regression set
    manifest = json.loads(MANIFEST_PATH.read_text()) if core else []
    if core:
        selected = select_core_emails(conn, since, {email['id'] for email in manifest})
    else:
        selected = select_emails(conn, since, num)
    records = fetch_inbound_emails_by_ids(conn, selected)

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT DISTINCT ON (inbound_email_id) inbound_email_id, corrected_category
            FROM email_training_samples
            WHERE inbound_email_id = ANY(%s) AND corrected_category IS NOT NULL
            ORDER BY inbound_email_id, updated_at DESC
        """, (list(selected),))
        grok_categories = {row['inbound_email_id']: row['corrected_category'] for row in cursor.fetchall()}
    conn.close()

    print(f"Downloading {len(records)} emails...")
    s3_client = get_s3_client()

    for i, record in enumerate(records, 1):
        email_id = record['id']
        set_name, reason = selected[email_id]
        raw_content = download_email_content(s3_client, record['blob_key'])
        if not raw_content:
            continue

        try:
            parsed = parse_email(raw_content)
        except Exception as e:
            print(f"Error parsing email {email_id}: {e}")
            continue
        text = extract_text_for_classification(parsed)
        if not text:
            continue

        # Candidates the rule rejects are kept aside to look for missed repair quotes
        if set_name == REPAIR_SERVICE_QUOTE and reason != 'INF-5509 example' and not is_repair_service_quote(text):
            set_name = NEEDS_REVIEW

        email = {
            'id': email_id,
            'set': set_name,
            'source': reason,
            'expected': set_name if set_name == REPAIR_SERVICE_QUOTE else None,
            'grok_category': grok_categories.get(email_id),
            'production_category': record['category_label'],
            'created_at': str(record['created_at']),
            'subject': parsed.get('subject', ''),
            'parsed': parsed,
        }

        set_dir = REGRESSIONS_DIR / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        (set_dir / f"{email_id}.eml").write_bytes(raw_content)
        write_readable(set_dir / f"{email_id}.txt", email, text)

        del email['parsed']
        manifest.append(email)

        if i % 100 == 0:
            print(f"  {i}/{len(records)}")

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {len(manifest)} emails to {REGRESSIONS_DIR}/")
    for set_name in sorted({email['set'] for email in manifest}):
        print(f"  {set_name}: {sum(1 for email in manifest if email['set'] == set_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download regression emails')
    parser.add_argument('--since', type=str, default='2026-04-14',
                        help='Only use emails received on or after this date (YYYY-MM-DD, default: 2026-04-14)')
    parser.add_argument('-n', '--num', type=int, default=150,
                        help='Number of emails per negative sample (default: 150)')
    parser.add_argument('--core', action='store_true',
                        help=f'Append RFQ and vendor quote response samples to {PENDING_REVIEW}/ for labeling')
    args = parser.parse_args()

    if args.core and not MANIFEST_PATH.exists():
        print(f"No {MANIFEST_PATH}; run without --core first")
        sys.exit(1)
    if not args.core and REGRESSIONS_DIR.exists() and any(REGRESSIONS_DIR.iterdir()):
        print(f"{REGRESSIONS_DIR}/ is not empty; remove it first to re-download")
        sys.exit(1)

    extract_regressions(date.fromisoformat(args.since), args.num, core=args.core)
