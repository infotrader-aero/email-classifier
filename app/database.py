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


def fetch_inbound_emails(conn, limit=None):
    """
    Fetch inbound emails with their ActiveStorage blob information.

    Returns a list of dictionaries with email metadata and blob keys.
    """
    query = """
        SELECT
            amie.id,
            amie.status,
            amie.message_id,
            amie.message_checksum,
            amie.created_at,
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
        ORDER BY amie.created_at DESC
    """

    if limit:
        query += f" LIMIT {limit}"

    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()


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
