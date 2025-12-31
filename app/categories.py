"""
Email categories for classification.
"""

import re

# Main email categories
REQUEST_FOR_QUOTE = 'request_for_quote'
LISTING_SERVICE = 'listing_service'
PERSONAL = 'personal'
SPAM = 'spam'
VENDOR_RESPONSE = 'vendor_response'
VENDOR_PROMOTION = 'vendor_promotion'
ORDER_CONFIRMATION = 'order_confirmation'
SHIPPING_NOTIFICATION = 'shipping_notification'
NEWSLETTER = 'newsletter'
AUTOMATED = 'automated'
OTHER = 'other'

# All categories
CATEGORIES = [
    REQUEST_FOR_QUOTE,
    LISTING_SERVICE,
    PERSONAL,
    SPAM,
    VENDOR_RESPONSE,
    VENDOR_PROMOTION,
    ORDER_CONFIRMATION,
    SHIPPING_NOTIFICATION,
    NEWSLETTER,
    AUTOMATED,
    OTHER,
]

# Category descriptions for labeling guidance
CATEGORY_DESCRIPTIONS = {
    REQUEST_FOR_QUOTE: (
        "Direct emails requesting quotes for aircraft parts. May include part numbers, "
        "quantities, conditions, and requests for pricing/availability. "
        "The sender is ASKING us for a quote directly (not via a listing service)."
    ),
    LISTING_SERVICE: (
        "RFQs forwarded through listing/marketplace platforms like ILSMart, PartsBase, "
        "SkySelect, Locatory, etc. These are templated emails from the platform that "
        "forward customer RFQs. The actual customer details are embedded in the template."
    ),
    PERSONAL: (
        "Personal emails not related to business. Includes casual conversations, "
        "meeting requests, personal matters."
    ),
    SPAM: (
        "Unsolicited emails, phishing attempts, scams, or irrelevant marketing "
        "from non-aviation companies."
    ),
    VENDOR_RESPONSE: (
        "Responses from vendors about quotes, pricing, or availability. "
        "Not a request, but a reply to a previous inquiry. Usually has RE: in subject."
    ),
    VENDOR_PROMOTION: (
        "Unsolicited sales emails from aviation vendors offering their inventory. "
        "They are OFFERING parts to us, not requesting quotes. Includes 'hot parts', "
        "'sale', 'we have in stock', promotional inventory lists."
    ),
    ORDER_CONFIRMATION: (
        "Confirmations of orders placed, purchase order acknowledgments."
    ),
    SHIPPING_NOTIFICATION: (
        "Shipping updates, tracking information, delivery notifications."
    ),
    NEWSLETTER: (
        "Industry newsletters, company updates, regular mailings with unsubscribe links."
    ),
    AUTOMATED: (
        "Automated system emails, notifications, alerts, password resets."
    ),
    OTHER: (
        "Emails that don't fit other categories."
    ),
}

# Keywords strongly associated with each category
CATEGORY_KEYWORDS = {
    LISTING_SERVICE: [
        'ilsmart.com', 'ilsmart', 'sent via ilsmart',
        'partsbase.com', 'partsbase', 'partsbase quote request',
        'skyselect.com', 'skyselect', 'via skyselect',
        'locatory.com', 'locatory',
        'stockmarket.aero', 'stock market aero',
        'entry code:', 'fast quote',
        'quote.ilsmart.com', 'do not reply to this email',
    ],
    AUTOMATED: [
        'do not reply', 'noreply', 'no-reply', 'automated',
        'notification', 'alert', 'system message',
        'password reset', 'verify your email',
        'automatic reply', 'auto-reply', 'out of office',
        'daily removals', 'daily report', 'scheduled report',
        'data administrator', 'system administrator',
        'your quote is ready', 'quote confirmation',
        'prepared for', 'prepared by',
        'this is a quotation only', 'this quote is valid for',
    ],
    REQUEST_FOR_QUOTE: [
        'request for quote', 'rfq', 'quote request', 'quotation',
        'please quote', 'need pricing', 'need price', 'price and availability',
        'looking for', 'in search of', 'seeking',
        'can you quote', 'can you provide', 'do you have',
        'could you please provide', 'kindly provide', 'kindly quote',
        'send me a quote', 'send quote', 'quote for the following',
        'aog', 'urgent requirement', 'immediate need',
        'we have one need', 'we have a need', 'we need',
        'could you please advise', 'please advise if you have',
        'exchange offer', 'advise and assist',
        'dear officer on duty', 'officer on duty',
        'a quote is wanted', 'quote wanted',
        'kindly request', 'we kindly request',
    ],
    VENDOR_RESPONSE: [
        'in response to', 'per your request', 'as requested',
        'our quote', 'our pricing', 'we can offer',
        'attached quote', 'please find attached',
        'we have the attached', 'quote and ppwk attached',
        'we can only offer', 'let me know if that works',
        'quote attached', 'pricing attached',
        'we have available', 'unit available',
    ],
    VENDOR_PROMOTION: [
        'hot parts', 'hot list', 'sale is live', 'parts sale',
        'exclusive discounts', 'special offer', 'special pricing',
        'we have in stock', 'available for sale', 'ready for dispatch',
        'submit your po', 'send your po', 'immediate dispatch',
        'inventory available', 'stock available', 'in our inventory',
        'we are offering', 'we offer', 'we are pleased to offer',
        'discover our', 'explore our offerings', 'check out our',
        'greeting from', 'greetings from',
        'we are specializing', 'we specialize in',
        'odm', 'oem services', 'obm services',
    ],
    ORDER_CONFIRMATION: [
        'order confirmation', 'order received', 'order number',
        'purchase order', 'po number', 'po#',
        'order placed', 'order accepted', 'order acknowledged',
    ],
    SHIPPING_NOTIFICATION: [
        'shipped', 'tracking', 'delivery', 'shipment',
        'fedex', 'ups', 'dhl', 'usps',
        'awb', 'air waybill', 'waybill number',
        'out for delivery', 'delivered', 'in transit',
    ],
    NEWSLETTER: [
        'newsletter', 'subscribe', 'unsubscribe',
        'view in browser', 'weekly update', 'monthly update',
        'update your preferences', 'email preferences',
        'change how you receive', 'opt out', 'optout',
        'follow us on', 'connect with us',
    ],
    SPAM: [
        'act now', 'limited time', 'free gift',
        'click here to win', 'congratulations',
        'million dollars', 'nigerian prince',
        'verify your account immediately',
        'dear madam/sir', 'dear sir/madam',
    ],
    PERSONAL: [
        'lunch', 'dinner', 'meeting tomorrow',
        'vacation', 'birthday', 'congratulations on',
        'how are you', 'hope you are well',
        'thanks for your help', 'appreciate',
        'happy new year', 'happy holidays', 'merry christmas',
        'season\'s greetings', 'wishing you a', 'best wishes',
        'hope this finds you well', 'looking forward to',
    ],
}


def get_category_from_keywords(text, from_address=''):
    """
    Heuristic classification based on keyword matching.

    Returns the most likely category based on keyword counts.
    Useful for initial labeling or as a baseline.
    """
    text_lower = text.lower()
    from_lower = from_address.lower() if from_address else ''

    # Priority rules - these override keyword scoring

    # 1. Listing service detection (ILSMart, PartsBase, SkySelect, etc.)
    listing_services = ['ilsmart', 'partsbase', 'skyselect', 'locatory', 'stockmarket.aero']
    if any(svc in from_lower or svc in text_lower for svc in listing_services):
        return LISTING_SERVICE

    # 2. Reply/Forward detection - check BEFORE outbound quotes (replies often quote original)
    if re.match(r'^(re|fw|fwd):\s', text_lower):
        return VENDOR_RESPONSE

    # 3. Outbound quote confirmations (system-generated quotes we sent)
    outbound_quote_signals = [
        'your quote is ready', 'quote confirmation',
        'this is a quotation only', 'this quote is valid for',
    ]
    if any(signal in text_lower for signal in outbound_quote_signals):
        return AUTOMATED

    # 4. Recall/system messages (Outlook recall feature)
    if text_lower.startswith('recall:'):
        return AUTOMATED

    # 5. Automated email detection (noreply addresses)
    if ('noreply@' in from_lower or 'no-reply@' in from_lower or 'donotreply@' in from_lower):
        return AUTOMATED

    # 6. Newsletter detection (unsubscribe is a strong signal)
    if 'unsubscribe' in text_lower or 'opt out' in text_lower or 'optout' in text_lower:
        return NEWSLETTER

    # 7. Personal/greeting emails
    personal_signals = [
        'happy new year', 'happy holidays', 'merry christmas',
        'season\'s greetings', 'wishing you', 'best wishes for',
        'happy birthday',
    ]
    if any(signal in text_lower for signal in personal_signals):
        return PERSONAL

    # 8. Vendor promotion detection (offering products TO recipient)
    vendor_promo_signals = [
        'hot parts', 'sale is live', 'exclusive discounts',
        'discover our', 'explore our', 'we are specializing',
        'greeting from', 'dear madam/sir', 'dear sir/madam',
        'submit your po', 'we have in stock',
    ]
    promo_score = sum(1 for signal in vendor_promo_signals if signal in text_lower)
    if promo_score >= 2:
        return VENDOR_PROMOTION

    # Standard keyword scoring
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return OTHER

    # Return category with highest score
    return max(scores, key=scores.get)
