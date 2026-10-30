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
VENDOR_QUOTE_RESPONSE = 'vendor_quote_response'
REPAIR_SERVICE_QUOTE = 'repair_service_quote'
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
    VENDOR_QUOTE_RESPONSE,
    REPAIR_SERVICE_QUOTE,
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
    VENDOR_QUOTE_RESPONSE: (
        "Responses from vendors to our request_for_quotes. These are quotes or pricing "
        "replies we receive back after sending an RFQ to a vendor. The vendor is answering "
        "our specific inquiry with pricing, availability, lead time, or terms."
    ),
    REPAIR_SERVICE_QUOTE: (
        "A repair station / MRO / shop quoting repair SERVICES on a part (RP repair, "
        "OH overhaul, BC bench check, TS test) rather than the sale of a unit. Includes "
        "ILS 'Quote Received' notices with a Service column and shop replies to our "
        "repair quote requests, often with TAT, evaluation fee, warranty or BER rate."
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
    VENDOR_QUOTE_RESPONSE: [
        # Explicit RFQ references
        'your rfq', 'quote for rfq', 'in response to your rfq',
        'per your rfq', 'replying to your rfq', 'reference your rfq',
        'regarding your rfq', 'for rfq #', 'for rfq#',
        # Formal quote language
        'in reply to your request for quote', 'as per your request for quote',
        'we are pleased to quote', 'please find our quote',
        'we are pleased to offer', 'attached quotation',
        'our quotation for', 'review our attached quotation',
        'please see our offer attached', 'attached offer',
        'please find attached file as our quotation',
        'kindly review our quote', 'kindly consider our quote',
        'thank you for your request', 'thank you for contacting us',
        # Offer/availability language
        'we have the unit', 'we can offer',
        'quote attached', 'offer attached',
        # No-quote / no-bid responses
        'no quote', 'no bid', 'no stock', 'have to no bid',
        'nq on all',
    ],
    # REPAIR_SERVICE_QUOTE has no keywords: only is_repair_service_quote assigns it,
    # since its wording also appears in our outbound repair quote requests
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
        # Patterns from post-processing (surplus/sale offers)
        'surplus for sale', 'parts for sale', 'inventory for sale',
        'stock for sale', 'we have a list of surplus',
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


# Repair service quote detection. Repair quotes look like vendor quotes (IQs),
# so these only match service-specific structure, never bare condition codes:
# "OH $5,000" in a vendor quote is an overhauled unit for sale, not a service.

# ILS "Quote Received" notice with one service code per quote ("Service: RP").
# Our outbound ILS RFQ form lists several ("Service: OH, RP, BC") and is a request.
ILS_SERVICE_QUOTE_PATTERN = re.compile(
    r'you have received a quote.*?service:\s*(?:rp|oh|bc|ts)\b(?!\s*,)', re.S
)

# Reply/forward to our outbound repair RFQs ("<Tenant> - Request for Repair Quote - PN X",
# "Please provide average Test, Repair and Overhaul pricing for PN X")
REPAIR_RFQ_REPLY_PATTERN = re.compile(
    r'^(?:re|fw|fwd):\s.*(?:request for repair quot|provide average test, repair and overhaul pricing)'
)

# A service label followed by a price ("Bench Test: $400", "Repair Average: $5000",
# "Test/Inspect: $330", "Overhaul - $9500.00", "BC:$604.00"). BC is the only
# service code allowed here: OH/RP/SV double as part condition codes.
SERVICE_PRICE_PATTERN = re.compile(
    r'\b(bench\s*(?:test|check)|bc|test(?:/(?:inspect\w*|eval\w*))?|repair|overhaul)'
    r'(?:\s*(?:average|avg\.?))?(?:\s*(?:average\s*)?(?:cost|price))?'
    r'\s*[:\-–]\s*(?:usd\s*)?\$\s*\d'
)

# Service price table: two service columns followed by a price
# ("Average Overhaul Average Repair Bench Test ... $1,075", "BENCH CHECK TAT AVG REP TAT AVG OH TAT ... $550")
SERVICE_COLUMN = r'(?:bench (?:check|test)|av(?:g\.?|erage) (?:rep\w*|oh|overhaul))'
SERVICE_PRICE_HEADER_PATTERN = re.compile(rf'{SERVICE_COLUMN}.{{0,40}}{SERVICE_COLUMN}.{{0,200}}\$\s*\d')

# Repair quote terms that confirm a single priced service
SERVICE_TERMS_PATTERN = re.compile(
    r'\b(?:eval(?:uation)?\s*(?:/\s*ber\s*)?fee|teardown fee|ber(?:/scrap)?\s*rate|warranty period|tat)\b'
)


REPLY_PREFIX_PATTERN = re.compile(r'^re:\s')

# Start of the quoted history in a reply ("From: X Sent: ...", "On <date> X wrote:")
QUOTED_HISTORY_PATTERN = re.compile(
    r'\bfrom:\*?\s.{0,200}?\b(?:sent|date):|\bon\s.{0,150}?\bwrote:|-{2,}\s*original message|\boriginal message:'
)

# A reply that answers our repair quote request: prices (with or without "$"),
# pricing in an attachment or table, or a capability decline
REPAIR_REPLY_ANSWER_PATTERN = re.compile(
    r'\$\s*\d|\b(?:test|tst|bc|rp|rep|repair|oh|ovh|overhaul)\s*[:\-]?\s*\d{3,}'
    r'|\b(?:pricing|prices|quote|quotation|rfq)\s+(?:below|attached)|\b(?:see|find) (?:the )?attached'
    r'|\bhere (?:is|are) (?:the |our )?(?:average )?(?:pricing|prices|quote)'
    r'|\bno cap(?:e?s|abilit\w*)\b|\b(?:not?|any) (?:have |hold )?(?:the )?(?:repair )?cap(?:e?s|abilit\w*)\b'
    r'|\bno quote\b|\bunable to\b|\b(?:can ?not|can\'t|cannot|do not|does not) (?:do|repair|overhaul|support|quote)'
)


# An outright sale priced alongside the services ("Outright: ... Price: $11,995")
OUTRIGHT_SALE_PATTERN = re.compile(r'\boutright\b.{0,400}?\$\s*\d')


def _latest_message(text_lower):
    """Strip the quoted history from a reply, keeping the subject and newest message."""
    match = QUOTED_HISTORY_PATTERN.search(text_lower)
    return text_lower[:match.start()] if match else text_lower


def is_repair_service_quote(text):
    """
    Check whether an email quotes repair services (RP/OH/BC/TS) on a part.

    An email that also quotes the outright sale of a unit is a vendor quote
    response, which covers both its part and service lines.
    """
    text_lower = text.lower()

    # A reply is judged by its newest message: the quoted repair quote below a
    # "thanks" or a follow-up question doesn't make the reply a quote. Forwards
    # keep the quoted part, since the forwarded email is the quote.
    if REPLY_PREFIX_PATTERN.match(text_lower):
        text_lower = _latest_message(text_lower)

    return _quotes_repair_services(text_lower) and not OUTRIGHT_SALE_PATTERN.search(text_lower)


def _quotes_repair_services(text_lower):
    if REPLY_PREFIX_PATTERN.match(text_lower):
        if REPAIR_RFQ_REPLY_PATTERN.search(text_lower) and REPAIR_REPLY_ANSWER_PATTERN.search(text_lower):
            return True
    elif REPAIR_RFQ_REPLY_PATTERN.search(text_lower):
        return True

    if ILS_SERVICE_QUOTE_PATTERN.search(text_lower):
        return True
    if SERVICE_PRICE_HEADER_PATTERN.search(text_lower):
        return True

    # Two priced services, or one priced service plus a repair quote term
    services = {
        re.sub(r'\s+', ' ', match).replace('check', 'test').replace('bc', 'bench test').split('/')[0]
        for match in SERVICE_PRICE_PATTERN.findall(text_lower)
    }
    return len(services) >= 2 or bool(services and SERVICE_TERMS_PATTERN.search(text_lower))


def get_category_from_keywords(text, from_address=''):
    """
    Heuristic classification based on keyword matching.

    Returns the most likely category based on keyword counts.
    Useful for initial labeling or as a baseline.
    """
    text_lower = text.lower()
    from_lower = from_address.lower() if from_address else ''

    # Priority rules - these override keyword scoring

    # 0. Repair service quote detection - BEFORE listing service, since ILS
    #    "Quote Received" repair notices come from ilsmart.com
    if is_repair_service_quote(text):
        return REPAIR_SERVICE_QUOTE

    # 1. Listing service detection (ILSMart, PartsBase, SkySelect, etc.)
    listing_services = ['ilsmart', 'partsbase', 'skyselect', 'locatory', 'stockmarket.aero']
    if any(svc in from_lower or svc in text_lower for svc in listing_services):
        return LISTING_SERVICE

    # 2. Vendor quote response detection (vendor replying to our RFQ)
    vendor_quote_response_signals = [
        'your rfq', 'quote for rfq', 'in response to your rfq',
        'per your rfq', 'regarding your rfq', 'replying to your rfq',
        'for rfq #', 'for rfq#',
        'in reply to your request for quote', 'as per your request for quote',
        'we are pleased to quote', 'please find our quote',
        'we are pleased to offer', 'attached quotation',
        'our quotation for', 'review our attached quotation',
        'please see our offer attached', 'attached offer',
        'please find attached file as our quotation',
        'kindly review our quote', 'kindly consider our quote',
        'thank you for your request', 'thank you for contacting us',
        'we have the unit', 'we can offer',
        'quote attached', 'offer attached',
        'no quote', 'no bid', 'no stock', 'have to no bid',
        'nq on all',
    ]
    if any(signal in text_lower for signal in vendor_quote_response_signals):
        return VENDOR_QUOTE_RESPONSE

    # 3. Reply/forward to our outbound RFQ (subject: "<Tenant> Request for Quote")
    if re.match(r'^(re|fw|fwd):\s', text_lower) and 'request for quote' in text_lower:
        return VENDOR_QUOTE_RESPONSE

    # 4. Reply/Forward detection - check BEFORE outbound quotes (replies often quote original)
    if re.match(r'^(re|fw|fwd):\s', text_lower):
        return VENDOR_RESPONSE

    # 5. Outbound quote confirmations (system-generated quotes we sent)
    outbound_quote_signals = [
        'your quote is ready', 'quote confirmation',
        'this is a quotation only', 'this quote is valid for',
    ]
    if any(signal in text_lower for signal in outbound_quote_signals):
        return AUTOMATED

    # 6. Recall/system messages (Outlook recall feature)
    if text_lower.startswith('recall:'):
        return AUTOMATED

    # 7. Automated email detection (noreply addresses)
    if ('noreply@' in from_lower or 'no-reply@' in from_lower or 'donotreply@' in from_lower):
        return AUTOMATED

    # 8. Newsletter detection (unsubscribe is a strong signal)
    if 'unsubscribe' in text_lower or 'opt out' in text_lower or 'optout' in text_lower:
        return NEWSLETTER

    # 9. Personal/greeting emails
    personal_signals = [
        'happy new year', 'happy holidays', 'merry christmas',
        'season\'s greetings', 'wishing you', 'best wishes for',
        'happy birthday',
    ]
    if any(signal in text_lower for signal in personal_signals):
        return PERSONAL

    # 10. Vendor promotion detection (offering products TO recipient)
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
