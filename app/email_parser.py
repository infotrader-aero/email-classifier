import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import re


def parse_email(raw_content):
    """
    Parse raw email content into structured data.

    Args:
        raw_content: Raw email bytes

    Returns:
        Dictionary with parsed email fields
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw_content)

    return {
        'subject': msg.get('Subject', ''),
        'from': msg.get('From', ''),
        'to': msg.get('To', ''),
        'cc': msg.get('Cc', ''),
        'date': msg.get('Date', ''),
        'body_text': get_text_content(msg),
        'body_html': get_html_content(msg),
        'has_attachments': has_attachments(msg),
        'attachment_names': get_attachment_names(msg),
    }


def get_text_content(msg):
    """Extract plain text content from email."""
    body = msg.get_body(preferencelist=('plain',))
    if body:
        try:
            return body.get_content()
        except Exception:
            return ''
    return ''


def get_html_content(msg):
    """Extract HTML content from email."""
    body = msg.get_body(preferencelist=('html',))
    if body:
        try:
            return body.get_content()
        except Exception:
            return ''
    return ''


def has_attachments(msg):
    """Check if email has attachments."""
    for part in msg.walk():
        if part.get_content_disposition() == 'attachment':
            return True
    return False


def get_attachment_names(msg):
    """Get list of attachment filenames."""
    attachments = []
    for part in msg.walk():
        if part.get_content_disposition() == 'attachment':
            filename = part.get_filename()
            if filename:
                attachments.append(filename)
    return attachments


def extract_text_for_classification(parsed_email):
    """
    Extract and clean text from parsed email for classification.

    Combines subject and body, removes HTML tags, normalizes whitespace.
    """
    text_parts = []

    # Add subject
    if parsed_email.get('subject'):
        text_parts.append(parsed_email['subject'])

    # Add body text (prefer plain text, fall back to HTML)
    body = parsed_email.get('body_text', '')
    if not body and parsed_email.get('body_html'):
        # Strip HTML tags
        soup = BeautifulSoup(parsed_email['body_html'], 'html.parser')
        body = soup.get_text(separator=' ')

    if body:
        text_parts.append(body)

    # Combine and clean
    text = ' '.join(text_parts)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()

    return text


def extract_features(parsed_email):
    """
    Extract features for classification.

    Returns a dictionary of features useful for ML classification.
    """
    text = extract_text_for_classification(parsed_email)
    text_lower = text.lower()
    subject = parsed_email.get('subject', '')
    subject_lower = subject.lower()

    # Aviation/parts related keywords
    aviation_keywords = [
        'part number', 'p/n', 'pn:', 'part no',
        'aircraft', 'aviation', 'aerospace',
        'quote', 'rfq', 'request for quote',
        'price', 'pricing', 'availability',
        'condition', 'serviceable', 'overhauled', 'as removed',
        'certification', '8130', 'easa', 'faa',
        'trace', 'traceability',
        'qty', 'quantity', 'unit',
        'lead time', 'delivery',
        'boeing', 'airbus', 'bombardier', 'embraer',
        'engine', 'apu', 'landing gear',
    ]

    # Newsletter/promotional indicators
    newsletter_keywords = [
        'unsubscribe', 'opt out', 'optout',
        'update your preferences', 'view in browser',
        'follow us on', 'connect with us',
    ]

    # Vendor promotion indicators
    vendor_promo_keywords = [
        'hot parts', 'sale is live', 'exclusive discounts',
        'discover our', 'explore our', 'we are specializing',
        'greeting from', 'we have in stock', 'submit your po',
        'dear madam/sir', 'dear sir/madam',
    ]

    # Personal email indicators
    personal_keywords = [
        'lunch', 'meeting', 'vacation', 'birthday',
        'thank you for', 'thanks for your help',
        'how are you', 'hope you are well',
        'personal', 'family', 'weekend',
    ]

    # Detect if this is a reply/forward
    is_reply = (
        subject_lower.startswith('re:') or
        subject_lower.startswith('fw:') or
        subject_lower.startswith('fwd:')
    )

    return {
        'text': text,
        'text_length': len(text),
        'subject': subject,
        'from_address': parsed_email.get('from', ''),
        'has_attachments': parsed_email.get('has_attachments', False),
        'attachment_count': len(parsed_email.get('attachment_names', [])),
        'is_reply': is_reply,
        'aviation_keyword_count': sum(1 for kw in aviation_keywords if kw in text_lower),
        'newsletter_keyword_count': sum(1 for kw in newsletter_keywords if kw in text_lower),
        'vendor_promo_keyword_count': sum(1 for kw in vendor_promo_keywords if kw in text_lower),
        'personal_keyword_count': sum(1 for kw in personal_keywords if kw in text_lower),
        'has_part_number_pattern': bool(re.search(r'\b[A-Z0-9]{3,}-?[A-Z0-9]{2,}\b', text)),
        'has_price_pattern': bool(re.search(r'\$[\d,]+\.?\d*|\bUSD\b|\bprice\b', text_lower)),
        'has_quantity_pattern': bool(re.search(r'\bqty\b|\bquantity\b|\bpcs\b|\beach\b', text_lower)),
        'has_unsubscribe': 'unsubscribe' in text_lower,
    }
