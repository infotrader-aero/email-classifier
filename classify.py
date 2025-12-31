#!/usr/bin/env python3
"""
Email Classification Script

Classifies a single email or batch of emails using a trained model.

Usage:
    python classify.py --model PATH --email-id ID
    python classify.py --model PATH --text "email content..."
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from app.classifier import EmailClassifier
from app.database import get_connection, fetch_inbound_emails
from app.storage import get_s3_client, download_email_content
from app.email_parser import parse_email, extract_text_for_classification
from app.categories import REQUEST_FOR_QUOTE


def classify_by_id(classifier, email_id):
    """Classify an email by its database ID."""
    conn = get_connection()

    query = """
        SELECT
            amie.id,
            amie.message_id,
            asb.key as blob_key
        FROM action_mailbox_inbound_emails amie
        JOIN active_storage_attachments asa
            ON asa.record_type = 'ActionMailbox::InboundEmail'
            AND asa.record_id = amie.id
            AND asa.name = 'raw_email'
        JOIN active_storage_blobs asb
            ON asb.id = asa.blob_id
        WHERE amie.id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (email_id,))
        record = cursor.fetchone()

    conn.close()

    if not record:
        print(f"Email not found: {email_id}")
        return None

    s3_client = get_s3_client()
    raw_content = download_email_content(s3_client, record['blob_key'])

    if not raw_content:
        print(f"Could not download email content")
        return None

    parsed = parse_email(raw_content)
    text = extract_text_for_classification(parsed)

    return classify_text(classifier, text, parsed)


def classify_text(classifier, text, parsed=None):
    """Classify email by text content."""
    prediction = classifier.predict(text)
    probabilities = classifier.predict_proba(text)

    is_rfq, rfq_confidence = classifier.is_request_for_quote(text)

    result = {
        'prediction': prediction,
        'is_request_for_quote': is_rfq,
        'rfq_confidence': rfq_confidence,
        'probabilities': probabilities,
    }

    if parsed:
        result['subject'] = parsed.get('subject', '')
        result['from'] = parsed.get('from', '')

    return result


def main():
    parser = argparse.ArgumentParser(description='Classify emails')
    parser.add_argument('--model', default='models/email_classifier.pkl',
                        help='Path to trained model pickle file')
    parser.add_argument('--email-id', type=int, help='Database ID of email to classify')
    parser.add_argument('--text', help='Raw text to classify')

    args = parser.parse_args()

    # Load classifier
    print(f"Loading model from: {args.model}")
    classifier = EmailClassifier.load(args.model)

    if args.email_id:
        result = classify_by_id(classifier, args.email_id)
    elif args.text:
        result = classify_text(classifier, args.text)
    else:
        parser.error("Either --email-id or --text is required")
        return

    if result:
        print("\n=== Classification Result ===")
        print(f"Prediction: {result['prediction']}")
        print(f"Is RFQ: {result['is_request_for_quote']} (confidence: {result['rfq_confidence']:.2%})")

        if 'subject' in result:
            print(f"Subject: {result['subject']}")
        if 'from' in result:
            print(f"From: {result['from']}")

        print("\nProbabilities:")
        for category, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            print(f"  {category}: {prob:.2%}")


if __name__ == '__main__':
    main()
