#!/usr/bin/env python3
"""
Extract example emails for manual review of classification accuracy.
"""

import os
import sys
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.database import get_connection, fetch_inbound_emails
from app.storage import get_s3_client, download_email_content
from app.email_parser import parse_email, extract_text_for_classification
from app.classifier import EmailClassifier


def load_classifier():
    """Load the trained classifier."""
    model_path = Path('models/email_classifier.pkl')
    if not model_path.exists():
        print("Error: No trained model found. Run train.py first.")
        sys.exit(1)

    return EmailClassifier.load(model_path)


def extract_examples(num_examples=100, category_filter=None):
    """
    Extract example emails and save them for review.

    Args:
        num_examples: Number of examples to extract
        category_filter: Optional category to filter by (e.g., 'request_for_quote')
    """
    # Create examples directory
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)

    # Clear existing examples
    for f in examples_dir.glob('*.txt'):
        f.unlink()
    for f in examples_dir.glob('*.json'):
        f.unlink()

    # Load classifier
    classifier = load_classifier()

    print(f"Connecting to database...")
    conn = get_connection()

    print(f"Connecting to S3...")
    s3_client = get_s3_client()

    print(f"Fetching emails...")
    emails = fetch_inbound_emails(conn, limit=1000)
    print(f"Found {len(emails)} emails")

    # Process emails and classify
    processed = []
    for email_record in emails:
        try:
            raw_content = download_email_content(s3_client, email_record['blob_key'])
            if not raw_content:
                continue

            parsed = parse_email(raw_content)
            text = extract_text_for_classification(parsed)

            if not text or len(text) < 20:
                continue

            # Classify
            prediction = classifier.predict(text)
            proba_dict = classifier.predict_proba(text)
            confidence = max(proba_dict.values())

            processed.append({
                'id': email_record['id'],
                'message_id': email_record['message_id'],
                'created_at': str(email_record['created_at']),
                'subject': parsed.get('subject', ''),
                'from': parsed.get('from', ''),
                'to': parsed.get('to', ''),
                'text': text,
                'prediction': prediction,
                'confidence': float(confidence),
                'has_attachments': parsed.get('has_attachments', False),
                'attachment_names': parsed.get('attachment_names', []),
            })
        except Exception as e:
            print(f"Error processing email {email_record['id']}: {e}")
            continue

    conn.close()
    print(f"Processed {len(processed)} emails")

    # Filter by category if specified
    if category_filter:
        processed = [e for e in processed if e['prediction'] == category_filter]
        print(f"Filtered to {len(processed)} emails with category '{category_filter}'")

    # Randomly sample if we have more than requested
    if len(processed) > num_examples:
        processed = random.sample(processed, num_examples)

    # Group by category for summary
    by_category = {}
    for email in processed:
        cat = email['prediction']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(email)

    print(f"\nCategory distribution in sample:")
    for cat, emails in sorted(by_category.items()):
        print(f"  {cat}: {len(emails)}")

    # Save individual examples
    for i, email in enumerate(processed, 1):
        # Create readable text file
        txt_path = examples_dir / f"{i:03d}_{email['prediction']}_{email['confidence']:.2f}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"EMAIL #{i}\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"ID: {email['id']}\n")
            f.write(f"Predicted Category: {email['prediction']}\n")
            f.write(f"Confidence: {email['confidence']:.2%}\n")
            f.write(f"Date: {email['created_at']}\n")
            f.write(f"\n" + "-" * 40 + "\n\n")
            f.write(f"From: {email['from']}\n")
            f.write(f"To: {email['to']}\n")
            f.write(f"Subject: {email['subject']}\n")
            f.write(f"Has Attachments: {email['has_attachments']}\n")
            if email['attachment_names']:
                f.write(f"Attachments: {', '.join(email['attachment_names'])}\n")
            f.write(f"\n" + "-" * 40 + "\n")
            f.write(f"BODY:\n" + "-" * 40 + "\n\n")
            f.write(email['text'])
            f.write("\n\n" + "=" * 80 + "\n")

    # Save summary JSON
    summary_path = examples_dir / '_summary.json'
    summary = {
        'total_examples': len(processed),
        'category_filter': category_filter,
        'by_category': {cat: len(emails) for cat, emails in by_category.items()},
        'emails': processed
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved {len(processed)} examples to {examples_dir}/")
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract email examples for review')
    parser.add_argument('-n', '--num', type=int, default=100,
                        help='Number of examples to extract (default: 100)')
    parser.add_argument('-c', '--category', type=str, default=None,
                        help='Filter by category (e.g., request_for_quote)')

    args = parser.parse_args()

    extract_examples(num_examples=args.num, category_filter=args.category)
