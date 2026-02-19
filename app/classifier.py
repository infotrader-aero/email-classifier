import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from .categories import CATEGORIES, REQUEST_FOR_QUOTE, LISTING_SERVICE, AUTOMATED, VENDOR_RESPONSE


# High-confidence patterns that override ML predictions
LISTING_SERVICE_PATTERNS = [
    'ilsmart', 'partsbase', 'skyselect', 'locatory', 'stockmarket.aero',
    'sent via ilsmart', 'via partsbase', 'via skyselect',
]

OUTBOUND_QUOTE_PATTERNS = [
    'your quote is ready', 'this is a quotation only', 'this quote is valid for',
]

# Patterns indicating a vendor is responding to our RFQ
VENDOR_RESPONSE_PATTERNS = [
    'your rfq',
    'quote for rfq',
]

# Regex to find dollar amounts (e.g., $100, $1,000, $3,985.00)
PRICE_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?')


def _contains_price_above_threshold(text, threshold=100):
    """Check if text contains a dollar amount above the threshold."""
    matches = PRICE_PATTERN.findall(text)
    for match in matches:
        # Remove $ and commas, then convert to float
        amount = float(match.replace('$', '').replace(',', ''))
        if amount > threshold:
            return True
    return False


class EmailClassifier:
    """
    Email classifier for categorizing inbound emails.

    Uses TF-IDF vectorization with Multinomial Naive Bayes classifier.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95,
            )),
            ('classifier', MultinomialNB(alpha=0.1)),
        ])
        self.is_trained = False
        self.categories = CATEGORIES

    def train(self, texts, labels):
        """
        Train the classifier.

        Args:
            texts: List of email text content
            labels: List of category labels
        """
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def _postprocess(self, text, prediction):
        """
        Apply rule-based overrides for high-confidence patterns.

        Some patterns (like listing service emails) are so distinctive
        that we override the ML prediction to ensure accuracy.
        """
        text_lower = text.lower()

        # 1. Check for reply/forward - these are responses, not new requests
        if re.match(r'^(re|fw|fwd):\s', text_lower):
            return VENDOR_RESPONSE

        # 2. Check for vendor response patterns (e.g., "your rfq" means they're responding to our RFQ)
        if any(pattern in text_lower for pattern in VENDOR_RESPONSE_PATTERNS):
            return VENDOR_RESPONSE

        # 3. Check for price > $100 - RFQs request prices, they don't include them
        if _contains_price_above_threshold(text):
            return VENDOR_RESPONSE

        # 4. Check for listing service patterns
        if any(pattern in text_lower for pattern in LISTING_SERVICE_PATTERNS):
            return LISTING_SERVICE

        # 5. Check for outbound quote patterns (but not if it's a reply)
        if any(pattern in text_lower for pattern in OUTBOUND_QUOTE_PATTERNS):
            return AUTOMATED

        return prediction

    def predict(self, text):
        """
        Predict the category of an email.

        Args:
            text: Email text content

        Returns:
            Predicted category string
        """
        if not self.is_trained:
            raise ValueError("Classifier has not been trained")

        original_text = text
        if isinstance(text, str):
            text = [text]

        ml_prediction = self.pipeline.predict(text)[0]
        return self._postprocess(original_text if isinstance(original_text, str) else original_text[0], ml_prediction)

    def predict_proba(self, text):
        """
        Get probability distribution over categories.

        Args:
            text: Email text content

        Returns:
            Dictionary mapping categories to probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier has not been trained")

        if isinstance(text, str):
            text = [text]

        probs = self.pipeline.predict_proba(text)[0]
        classes = self.pipeline.classes_

        return dict(zip(classes, probs))

    def is_request_for_quote(self, text, threshold=0.5):
        """
        Check if email is a request for quote.

        Args:
            text: Email text content
            threshold: Probability threshold for positive classification

        Returns:
            Tuple of (is_rfq: bool, confidence: float)
        """
        probs = self.predict_proba(text)
        rfq_prob = probs.get(REQUEST_FOR_QUOTE, 0.0)

        return rfq_prob >= threshold, rfq_prob

    def evaluate(self, texts, labels):
        """
        Evaluate the classifier on test data.

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.pipeline.predict(texts)

        return {
            'classification_report': classification_report(labels, predictions),
            'confusion_matrix': confusion_matrix(labels, predictions),
            'accuracy': np.mean(predictions == np.array(labels)),
        }

    def cross_validate(self, texts, labels, cv=5):
        """
        Perform cross-validation.

        Returns:
            Dictionary with cross-validation scores
        """
        scores = cross_val_score(self.pipeline, texts, labels, cv=cv)

        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
        }

    def save(self, filepath):
        """Save the trained model to a pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'is_trained': self.is_trained,
                'categories': self.categories,
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load a trained model from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        classifier = cls()
        classifier.pipeline = data['pipeline']
        classifier.is_trained = data['is_trained']
        classifier.categories = data['categories']

        return classifier


def train_and_evaluate(texts, labels, test_size=0.2, random_state=42):
    """
    Train and evaluate a classifier with train/test split.

    Args:
        texts: List of email text content
        labels: List of category labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (classifier, evaluation_results)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    classifier = EmailClassifier()
    classifier.train(X_train, y_train)

    evaluation = classifier.evaluate(X_test, y_test)
    cv_results = classifier.cross_validate(X_train, y_train)

    return classifier, {
        'test_evaluation': evaluation,
        'cross_validation': cv_results,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
