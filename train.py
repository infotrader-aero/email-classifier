#!/usr/bin/env python3
"""
Email Classifier Training Script

This script:
1. Connects to PostgreSQL to fetch email metadata
2. Downloads email content from S3
3. Parses and extracts features from emails
4. Uses heuristic labeling for initial training data
5. Trains a classifier and saves it as a pickle file

Usage:
    python train.py [--limit N] [--output PATH]

Environment variables required:
    DATABASE_URL: PostgreSQL connection string
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_BUCKET: S3 bucket name (default: rfq-mail-production)
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.database import get_connection, fetch_inbound_emails, fetch_email_count
from app.storage import get_s3_client, download_emails_batch
from app.email_parser import parse_email, extract_text_for_classification
from app.categories import get_category_from_keywords, CATEGORIES, CATEGORY_DESCRIPTIONS
from app.classifier import EmailClassifier, train_and_evaluate


def load_emails(limit=None):
    """
    Load emails from database and S3.

    Returns:
        List of dictionaries with email data and parsed content
    """
    print("Connecting to database...")
    conn = get_connection()

    total_count = fetch_email_count(conn)
    print(f"Total emails in database: {total_count}")

    print(f"Fetching email metadata{f' (limit: {limit})' if limit else ''}...")
    email_records = fetch_inbound_emails(conn, limit=limit)
    print(f"Found {len(email_records)} emails with blob data")

    conn.close()

    print("Connecting to S3...")
    s3_client = get_s3_client()

    print("Downloading and parsing emails...")
    emails = []
    errors = 0

    for record, raw_content in download_emails_batch(s3_client, email_records):
        try:
            parsed = parse_email(raw_content)
            text = extract_text_for_classification(parsed)

            if text:  # Only include emails with extractable text
                emails.append({
                    'id': record['id'],
                    'message_id': record['message_id'],
                    'created_at': record['created_at'],
                    'parsed': parsed,
                    'text': text,
                })
        except Exception as e:
            errors += 1
            print(f"Error parsing email {record['id']}: {e}")

    print(f"Successfully parsed {len(emails)} emails ({errors} errors)")
    return emails


def label_emails_heuristically(emails):
    """
    Apply heuristic labeling based on keyword matching.

    This provides initial training labels that can be manually corrected.
    """
    print("Applying heuristic labels...")

    for email in emails:
        from_address = email.get('parsed', {}).get('from', '')
        email['label'] = get_category_from_keywords(email['text'], from_address)

    # Print distribution
    from collections import Counter
    distribution = Counter(email['label'] for email in emails)
    print("\nLabel distribution:")
    for category in CATEGORIES:
        count = distribution.get(category, 0)
        pct = (count / len(emails) * 100) if emails else 0
        print(f"  {category}: {count} ({pct:.1f}%)")

    return emails


def train_classifier(emails, output_path):
    """
    Train the classifier and save to pickle file.
    """
    if len(emails) < 10:
        print("Not enough emails for training (minimum 10 required)")
        return None

    texts = [email['text'] for email in emails]
    labels = [email['label'] for email in emails]

    print(f"\nTraining classifier on {len(texts)} emails...")

    # Check minimum samples per class
    from collections import Counter
    label_counts = Counter(labels)
    min_samples = min(label_counts.values())

    if min_samples < 2:
        print("Warning: Some categories have less than 2 samples.")
        print("Filtering to categories with at least 2 samples...")

        valid_labels = {label for label, count in label_counts.items() if count >= 2}
        filtered = [(t, l) for t, l in zip(texts, labels) if l in valid_labels]
        texts = [t for t, l in filtered]
        labels = [l for t, l in filtered]

        print(f"Training on {len(texts)} emails after filtering")

    if len(texts) < 10:
        print("Not enough valid samples for training")
        return None

    try:
        classifier, results = train_and_evaluate(texts, labels)

        print("\n=== Training Results ===")
        print(f"Training samples: {results['train_size']}")
        print(f"Test samples: {results['test_size']}")
        print(f"Test accuracy: {results['test_evaluation']['accuracy']:.2%}")
        print(f"\nCross-validation: {results['cross_validation']['mean']:.2%} "
              f"(+/- {results['cross_validation']['std']:.2%})")
        print("\nClassification Report:")
        print(results['test_evaluation']['classification_report'])

        # Save the model
        output_path = Path(output_path)
        classifier.save(output_path)
        print(f"\nModel saved to: {output_path}")

        return classifier

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_for_manual_labeling(emails, output_path):
    """
    Export emails to CSV for manual labeling.
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'message_id', 'created_at', 'subject', 'from',
            'text_preview', 'heuristic_label', 'manual_label'
        ])
        writer.writeheader()

        for email in emails:
            writer.writerow({
                'id': email['id'],
                'message_id': email['message_id'],
                'created_at': email['created_at'],
                'subject': email['parsed'].get('subject', '')[:200],
                'from': email['parsed'].get('from', ''),
                'text_preview': email['text'][:500],
                'heuristic_label': email['label'],
                'manual_label': '',  # To be filled manually
            })

    print(f"Exported {len(emails)} emails for labeling to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train email classifier')
    parser.add_argument('--limit', type=int, help='Limit number of emails to process')
    parser.add_argument('--output', default='models/email_classifier.pkl',
                        help='Output path for pickle file')
    parser.add_argument('--export-csv', help='Export emails to CSV for manual labeling')

    args = parser.parse_args()

    # Validate environment
    required_vars = ['DATABASE_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set the following environment variables:")
        print("  DATABASE_URL: PostgreSQL connection string")
        print("  AWS_ACCESS_KEY_ID: AWS access key")
        print("  AWS_SECRET_ACCESS_KEY: AWS secret key")
        print("  AWS_REGION: AWS region (optional, default: us-east-1)")
        print("  S3_BUCKET: S3 bucket name (optional, default: rfq-mail-production)")
        sys.exit(1)

    # Load and process emails
    emails = load_emails(limit=args.limit)

    if not emails:
        print("No emails found to process")
        sys.exit(1)

    # Apply heuristic labels
    emails = label_emails_heuristically(emails)

    # Export for manual labeling if requested
    if args.export_csv:
        export_for_manual_labeling(emails, args.export_csv)

    # Train classifier
    train_classifier(emails, args.output)


if __name__ == '__main__':
    main()
