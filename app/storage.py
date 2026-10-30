import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Create an S3 client using environment variables."""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )


def get_bucket_name():
    """Get the S3 bucket name from environment or default."""
    return os.environ.get('S3_BUCKET', 'rfq-mail-production')


def download_email_content(s3_client, blob_key, bucket=None):
    """
    Download email content from S3 using the blob key.

    Args:
        s3_client: boto3 S3 client
        blob_key: The ActiveStorage blob key
        bucket: Optional bucket name override

    Returns:
        Raw email content as bytes, or None if not found
    """
    bucket = bucket or get_bucket_name()

    try:
        response = s3_client.get_object(Bucket=bucket, Key=blob_key)
        return response['Body'].read()
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"Email not found in S3: {blob_key}")
            return None
        raise


def download_emails_batch(s3_client, email_records, bucket=None, max_workers=16):
    """
    Download multiple emails from S3 concurrently.

    Args:
        s3_client: boto3 S3 client (thread-safe)
        email_records: List of email records with blob_key
        bucket: Optional bucket name override
        max_workers: Number of concurrent downloads

    Yields:
        Tuples of (email_record, raw_content), in completion order
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_email_content, s3_client, record['blob_key'], bucket): record
            for record in email_records
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            # Drop the finished future so its email content can be freed once handled
            record = futures.pop(future)
            content = future.result()
            if content:
                yield record, content
            if i % 1000 == 0:
                print(f"  Downloaded {i}/{total}")
