import os
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse


def get_connection():
    """Create a database connection from DATABASE_URL environment variable."""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    parsed = urlparse(database_url)

    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path[1:],  # Remove leading slash
        user=parsed.username,
        password=parsed.password,
        cursor_factory=RealDictCursor
    )


INBOUND_EMAIL_BLOB_QUERY = """
    SELECT
        amie.id,
        amie.status,
        amie.message_id,
        amie.message_checksum,
        amie.created_at,
        amie.category_label,
        asb.key as blob_key,
        asb.filename,
        asb.content_type,
        asb.byte_size,
        asb.service_name
    FROM action_mailbox_inbound_emails amie
    JOIN active_storage_attachments asa
        ON asa.record_type = 'ActionMailbox::InboundEmail'
        AND asa.record_id = amie.id
        AND asa.name = 'raw_email'
    JOIN active_storage_blobs asb
        ON asb.id = asa.blob_id
"""


def fetch_inbound_emails(conn, limit=None, before=None):
    """
    Fetch inbound emails with their ActiveStorage blob information.

    Args:
        conn: Database connection
        limit: Optional max number of emails to return
        before: Optional datetime/date — only fetch emails created before this date

    Returns a list of dictionaries with email metadata and blob keys.
    """
    query = INBOUND_EMAIL_BLOB_QUERY

    params = []
    if before:
        query += " WHERE amie.created_at < %s"
        params.append(before)

    query += " ORDER BY amie.created_at DESC"

    if limit:
        query += f" LIMIT {limit}"

    with conn.cursor() as cursor:
        cursor.execute(query, params or None)
        return cursor.fetchall()


def fetch_inbound_emails_by_ids(conn, ids):
    """Fetch inbound emails with their ActiveStorage blob information by id."""
    if not ids:
        return []

    query = INBOUND_EMAIL_BLOB_QUERY + " WHERE amie.id = ANY(%s) ORDER BY amie.id"

    with conn.cursor() as cursor:
        cursor.execute(query, (list(ids),))
        return cursor.fetchall()


# Broad prefilter for repair service quotes. Matches are confirmed with
# categories.is_repair_service_quote on the parsed email text.
REPAIR_SERVICE_QUOTE_CANDIDATE_SQL = (
    r"(you have received a quote.*service:\s*(rp|oh|bc|ts)\M"
    r"|request for repair quot|provide average test, repair and overhaul pricing"
    r"|(bench|repair|overhaul|test)[^$]{0,40}\$\s*\d)"
)


def fetch_repair_service_quote_candidate_ids(conn, since=None):
    """
    Fetch ids of inbound emails that may be repair service quotes.

    Uses the subject and body stored in email_training_samples.

    Args:
        conn: Database connection
        since: Optional datetime/date — only emails created on or after this date
    """
    query = """
        SELECT DISTINCT ets.inbound_email_id
        FROM email_training_samples ets
        JOIN action_mailbox_inbound_emails amie ON amie.id = ets.inbound_email_id
        WHERE (ets.subject || ' ' || ets.body_text) ~* %s
    """
    params = [REPAIR_SERVICE_QUOTE_CANDIDATE_SQL]

    if since:
        query += " AND amie.created_at >= %s"
        params.append(since)

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        return sorted(row['inbound_email_id'] for row in cursor.fetchall())


def fetch_email_count(conn):
    """Get total count of inbound emails."""
    query = """
        SELECT COUNT(*) as count
        FROM action_mailbox_inbound_emails
    """

    with conn.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
        return result['count']


def fetch_corrected_labels(conn):
    """
    Fetch corrected category labels from email_training_samples.

    Returns a dictionary mapping inbound_email_id to corrected_category.
    Only includes records where corrected_category is set.
    """
    query = """
        SELECT
            inbound_email_id,
            corrected_category
        FROM email_training_samples
        WHERE corrected_category IS NOT NULL
          AND corrected_category != ''
    """

    with conn.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

    return {row['inbound_email_id']: row['corrected_category'] for row in results}
