#!/usr/bin/env python3
"""
Check a classifier change against the hand-verified regression set from extract_regressions.py.

Run once on the current classifier (e.g. on master) to save baseline predictions,
then on the change (new code, new model, or both) to compare:

    python check_regressions.py --save-baseline
    python check_regressions.py --model models/new_classifier.pkl

Each email's `expected` label in the manifest is its hand-verified category. Reports:
  - accuracy per category; RFQ and vendor quote response categories are critical
  - regressions: emails the baseline got right and the candidate gets wrong
  - improvements: emails the baseline got wrong and the candidate gets right
  - emails both get wrong (known issues, not failures)
  - changed predictions on emails without a label

Writes regressions/report.csv. Exits 1 on regressions or unlabeled changes.
"""

import csv
import sys
import json
import argparse
import warnings
from collections import Counter

from app.classifier import EmailClassifier
from app.email_parser import parse_email, extract_text_for_classification
from app.regressions import REGRESSIONS_DIR, MANIFEST_PATH, PENDING_REVIEW

REPORT_PATH = REGRESSIONS_DIR / 'report.csv'
BASELINE_PATH = REGRESSIONS_DIR / 'baseline.json'

# Sets kept for reference but not scored: unverified emails, ambiguous ones, and
# repair quotes (INF-5509, parked until there is more data)
UNCHECKED_SETS = {PENDING_REVIEW, 'unsure', 'repair_service_quote'}

# Categories that drive RFQ and incoming quote creation: these must not regress
CRITICAL_CATEGORIES = {'request_for_quote', 'listing_service', 'vendor_quote_response'}


def classify_manifest(model_path, manifest):
    """Classify every email in the manifest. Returns {email_id: category}."""
    # Pickles were saved with an older scikit-learn
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    classifier = EmailClassifier.load(model_path)

    predictions = {}
    for email in manifest:
        raw_content = (REGRESSIONS_DIR / email['set'] / f"{email['id']}.eml").read_bytes()
        text = extract_text_for_classification(parse_email(raw_content))
        predictions[email['id']] = classifier.predict(text)
    return predictions


def show(title, rows):
    print(f"\n{title}: {len(rows)}")
    for r in rows:
        expected = f" (expected {r['expected']})" if r['expected'] else ''
        print(f"  {r['set']}/{r['id']}.txt  {r['baseline']} -> {r['candidate']}{expected}  | {r['subject'][:80]}")


def main():
    parser = argparse.ArgumentParser(description='Check classifier regressions on the hand-verified set')
    parser.add_argument('--model', default='models/email_classifier.pkl', help='Model to classify with')
    parser.add_argument('--save-baseline', action='store_true',
                        help=f'Save predictions to {BASELINE_PATH} instead of comparing')
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"No {MANIFEST_PATH}; run extract_regressions.py first")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"Classifying {len(manifest)} emails with {args.model}...")
    predictions = classify_manifest(args.model, manifest)

    if args.save_baseline:
        with open(BASELINE_PATH, 'w', encoding='utf-8') as f:
            json.dump({str(email_id): category for email_id, category in predictions.items()}, f)
        print(f"Baseline saved to {BASELINE_PATH}")
        return

    if not BASELINE_PATH.exists():
        print(f"No {BASELINE_PATH}; run with --save-baseline on the current classifier first")
        sys.exit(1)

    with open(BASELINE_PATH, encoding='utf-8') as f:
        baseline = json.load(f)

    rows = [{**email, 'baseline': baseline.get(str(email['id'])), 'candidate': predictions[email['id']]}
            for email in manifest]
    checked = [r for r in rows if r['set'] not in UNCHECKED_SETS]
    labeled = [r for r in checked if r['expected']]
    regressions = [r for r in labeled if r['baseline'] == r['expected'] != r['candidate']]
    improvements = [r for r in labeled if r['baseline'] != r['expected'] == r['candidate']]
    still_wrong = [r for r in labeled if r['baseline'] != r['expected'] and r['candidate'] != r['expected']]
    unlabeled_changes = [r for r in checked if not r['expected'] and r['baseline'] != r['candidate']]

    accuracy = lambda key, items: sum(r[key] == r['expected'] for r in items) / len(items) if items else 0
    print(f"\nAccuracy on {len(labeled)} labeled emails: baseline {accuracy('baseline', labeled):.1%}, "
          f"candidate {accuracy('candidate', labeled):.1%}")

    print("\nAccuracy by expected category (* = critical):")
    for category in sorted({r['expected'] for r in labeled}, key=lambda c: (c not in CRITICAL_CATEGORIES, c)):
        items = [r for r in labeled if r['expected'] == category]
        marker = '*' if category in CRITICAL_CATEGORIES else ' '
        print(f"  {marker} {category:24s} {len(items):5d}   baseline {accuracy('baseline', items):6.1%}   "
              f"candidate {accuracy('candidate', items):6.1%}")

    show("CRITICAL regressions (RFQ / vendor quote response)",
         [r for r in regressions if r['expected'] in CRITICAL_CATEGORIES])
    show("Regressions (baseline right, candidate wrong)", regressions)
    show("Improvements (baseline wrong, candidate right)", improvements)
    print(f"\nStill wrong in both: {len(still_wrong)} "
          f"({dict(Counter(r['expected'] for r in still_wrong).most_common())})")
    show("Changed predictions without a label", unlabeled_changes)

    with open(REPORT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, extrasaction='ignore', fieldnames=[
            'id', 'set', 'source', 'expected', 'review_confidence', 'review_reason', 'tenant',
            'production_category', 'baseline', 'candidate', 'created_at', 'subject',
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReport saved to {REPORT_PATH}")

    sys.exit(1 if regressions or unlabeled_changes else 0)


if __name__ == '__main__':
    main()
