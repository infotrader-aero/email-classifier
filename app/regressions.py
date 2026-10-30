"""
Regression set shared by train.py, extract_regressions.py and check_regressions.py.
"""

import json
from pathlib import Path

REGRESSIONS_DIR = Path('regressions')
MANIFEST_PATH = REGRESSIONS_DIR / 'manifest.json'

# Every Nth repair service quote is held out of training for the regression set
REGRESSION_HOLDOUT_MODULUS = 3


def is_regression_holdout(email_id):
    """Check whether an email is held out of training for the regression set."""
    return email_id % REGRESSION_HOLDOUT_MODULUS == 0


def regression_ids():
    """Ids of every email in the downloaded regression set, kept out of training."""
    if not MANIFEST_PATH.exists():
        return set()
    with open(MANIFEST_PATH, encoding='utf-8') as f:
        return {email['id'] for email in json.load(f)}
