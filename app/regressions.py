"""
Regression set shared by extract_regressions.py and check_regressions.py.
"""

from pathlib import Path

REGRESSIONS_DIR = Path('regressions')
MANIFEST_PATH = REGRESSIONS_DIR / 'manifest.json'

# Emails awaiting a hand-verified label
PENDING_REVIEW = 'pending_review'
