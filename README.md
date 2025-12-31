# Email Classifier

Machine learning classifier for categorizing inbound emails from ActionMailbox.

Specifically designed for an aviation parts trading company to identify:
- **Request for Quote (RFQ)** - Emails requesting quotes for aircraft parts
- **Vendor Response** - Responses from vendors about pricing/availability
- **Order Confirmation** - Purchase order acknowledgments
- **Shipping Notification** - Tracking and delivery updates
- **Newsletter** - Industry newsletters and updates
- **Automated** - System notifications and alerts
- **Personal** - Non-business related emails
- **Spam** - Unsolicited/phishing emails
- **Other** - Emails that don't fit other categories

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `AWS_ACCESS_KEY_ID`: AWS access key for S3
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region (default: us-east-1)
- `S3_BUCKET`: S3 bucket name (default: rfq-mail-production)

## Training

Train the classifier using emails from the database:

```bash
# Train on all emails
python train.py

# Train on limited sample
python train.py --limit 1000

# Export emails for manual labeling
python train.py --limit 500 --export-csv data/emails_for_labeling.csv

# Specify output path
python train.py --output models/my_classifier.pkl
```

The training script:
1. Fetches email metadata from PostgreSQL
2. Downloads raw email content from S3
3. Parses emails and extracts text
4. Applies heuristic labels based on keyword matching
5. Trains a TF-IDF + Naive Bayes classifier
6. Saves the model as a pickle file

## Classification

Classify emails using a trained model:

```bash
# Classify by database ID
python classify.py --email-id 12345

# Classify raw text
python classify.py --text "Please quote part number 1234-5678, qty 2, serviceable condition"

# Use specific model
python classify.py --model models/custom_classifier.pkl --email-id 12345
```

## Integration with Rails

The trained pickle file can be loaded from a Rails background job to classify incoming emails:

```ruby
# Example integration (requires pycall gem or similar)
class ClassifyEmailJob < ApplicationJob
  def perform(inbound_email_id)
    # Load and classify using Python model
    result = EmailClassifierService.classify(inbound_email_id)

    if result[:is_request_for_quote]
      # Route to RFQ parsing job
      ParseRfqEmailJob.perform_later(inbound_email_id)
    end
  end
end
```

## Model Details

- **Vectorizer**: TF-IDF with 1-2 ngrams, max 5000 features
- **Classifier**: Multinomial Naive Bayes
- **Categories**: 9 email categories optimized for aviation parts trading

## Improving Accuracy

For better classification accuracy:

1. Export emails for manual labeling:
   ```bash
   python train.py --limit 1000 --export-csv data/emails.csv
   ```

2. Manually review and correct labels in the CSV

3. Implement a training script that loads manual labels:
   ```python
   # Load manually labeled data
   import pandas as pd
   df = pd.read_csv('data/emails_labeled.csv')
   labeled = df[df['manual_label'].notna()]
   ```

4. Retrain with corrected labels for improved accuracy
