#!/usr/bin/env python3
"""
Compare a baseline and a candidate model on the regression set from extract_regressions.py.

The baseline runs without the repair service quote rule, reproducing the
classifier before repair_service_quote existed. Each email's `expected` label
in the manifest is its hand-verified category. Reports:
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
from collections import Counter, defaultdict
from unittest import mock

from app.classifier import EmailClassifier
from app.email_parser import parse_email, extract_text_for_classification
from app.regressions import REGRESSIONS_DIR, MANIFEST_PATH

REPORT_PATH = REGRESSIONS_DIR / 'report.csv'

# Sets that are not checked for misses or false positives
UNCHECKED_SETS = {'needs_review', 'unsure', 'pending_review'}

# Categories that drive RFQ and incoming quote creation: these must not regress
CRITICAL_CATEGORIES = {'request_for_quote', 'listing_service', 'vendor_quote_response'}


def predict_all(classifier, texts, repair_rule=True):
    if repair_rule:
        return [classifier.predict(text) for text in texts]
    with mock.patch('app.classifier.is_repair_service_quote', return_value=False):
        return [classifier.predict(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(description='Check repair service quote regressions')
    parser.add_argument('--baseline', default='models/email_classifier_10-30-26.pkl.bak',
                        help='Baseline model, run without the repair rule')
    parser.add_argument('--candidate', default='models/email_classifier.pkl',
                        help='Candidate model')
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"No {MANIFEST_PATH}; run extract_regressions.py first")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding='utf-8') as f:
        manifest = json.load(f)

    texts = []
    for email in manifest:
        raw_content = (REGRESSIONS_DIR / email['set'] / f"{email['id']}.eml").read_bytes()
        texts.append(extract_text_for_classification(parse_email(raw_content)))

    # Pickles were saved with an older scikit-learn
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    print(f"Classifying {len(manifest)} emails...")
    baseline = predict_all(EmailClassifier.load(args.baseline), texts, repair_rule=False)
    candidate = predict_all(EmailClassifier.load(args.candidate), texts)

    rows = []
    for email, before, after in zip(manifest, baseline, candidate):
        rows.append({**email, 'baseline': before, 'candidate': after})

    checked = [r for r in rows if r['set'] not in UNCHECKED_SETS]
    labeled = [r for r in checked if r['expected']]
    regressions = [r for r in labeled if r['baseline'] == r['expected'] != r['candidate']]
    improvements = [r for r in labeled if r['baseline'] != r['expected'] == r['candidate']]
    still_wrong = [r for r in labeled if r['baseline'] != r['expected'] and r['candidate'] != r['expected']]
    unlabeled_changes = [r for r in checked if not r['expected'] and r['baseline'] != r['candidate']]

    by_set = defaultdict(list)
    for r in rows:
        by_set[r['set']].append(r)

    print(f"\nBaseline:  {args.baseline} (no repair rule)")
    print(f"Candidate: {args.candidate}")
    for set_name, set_rows in sorted(by_set.items()):
        print(f"\n== {set_name} ({len(set_rows)}) ==")
        print(f"  baseline:  {dict(Counter(r['baseline'] for r in set_rows).most_common())}")
        print(f"  candidate: {dict(Counter(r['candidate'] for r in set_rows).most_common())}")

    def show(title, items):
        print(f"\n{title}: {len(items)}")
        for r in items:
            expected = f" (expected {r['expected']})" if r['expected'] else ''
            print(f"  {r['set']}/{r['id']}.txt  {r['baseline']} -> {r['candidate']}{expected}  | {r['subject'][:80]}")

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
    show("Still wrong in both", still_wrong)
    show("Changed predictions without a label", unlabeled_changes)

    with open(REPORT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, extrasaction='ignore', fieldnames=[
            'id', 'set', 'source', 'expected', 'review_confidence', 'review_reason', 'grok_category', 'production_category',
            'baseline', 'candidate', 'created_at', 'subject',
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReport saved to {REPORT_PATH}")

    sys.exit(1 if regressions or unlabeled_changes else 0)


if __name__ == '__main__':
    main()
