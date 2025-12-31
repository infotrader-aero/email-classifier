#!/usr/bin/env python3
"""
Email Classifier FastAPI Server

Provides HTTP API for email classification with bearer token authentication.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000

Environment variables:
    API_KEY: Bearer token for authentication (required)
    MODEL_PATH: Path to pickle file (default: models/email_classifier.pkl)
"""

import os
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.classifier import EmailClassifier
from app.categories import CATEGORIES, CATEGORY_DESCRIPTIONS

app = FastAPI(
    title="Email Classifier API",
    description="Classify emails for aviation parts trading",
    version="1.0.0",
)

security = HTTPBearer()


def get_api_key():
    """Get API key from environment."""
    api_key = os.environ.get('API_KEY')
    if not api_key:
        raise ValueError("API_KEY environment variable is required")
    return api_key


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify bearer token matches API_KEY."""
    api_key = get_api_key()
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@lru_cache()
def get_classifier():
    """Load classifier once and cache it."""
    model_path = os.environ.get('MODEL_PATH', 'models/email_classifier.pkl')

    if not Path(model_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {model_path}. Run training first."
        )

    return EmailClassifier.load(model_path)


class ClassifyRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Please quote part number 1234-5678, qty 2, serviceable condition. Need price and availability."
            }
        }


class ClassifyResponse(BaseModel):
    category: str
    is_request_for_quote: bool
    is_listing_service: bool
    confidence: float
    probabilities: dict

    class Config:
        json_schema_extra = {
            "example": {
                "category": "request_for_quote",
                "is_request_for_quote": True,
                "is_listing_service": False,
                "confidence": 0.89,
                "probabilities": {
                    "request_for_quote": 0.89,
                    "vendor_response": 0.05,
                    "other": 0.06
                }
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class CategoriesResponse(BaseModel):
    categories: list
    descriptions: dict


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint (no auth required)."""
    model_path = os.environ.get('MODEL_PATH', 'models/email_classifier.pkl')
    model_loaded = Path(model_path).exists()

    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.get("/categories", response_model=CategoriesResponse, dependencies=[Depends(verify_token)])
def list_categories():
    """List all email categories."""
    return {
        "categories": CATEGORIES,
        "descriptions": CATEGORY_DESCRIPTIONS
    }


RFQ_CATEGORIES = ['request_for_quote', 'listing_service']


@app.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(verify_token)])
def classify_email(request: ClassifyRequest):
    """
    Classify an email and return the predicted category.

    Returns the most likely category, whether it's a request for quote,
    confidence score, and probability distribution across all categories.
    """
    classifier = get_classifier()

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    prediction = classifier.predict(request.text)
    probabilities = classifier.predict_proba(request.text)
    confidence = max(probabilities.values())

    return {
        "category": prediction,
        "is_request_for_quote": prediction in RFQ_CATEGORIES,
        "is_listing_service": prediction == 'listing_service',
        "confidence": confidence,
        "probabilities": probabilities
    }


@app.post("/batch-classify", dependencies=[Depends(verify_token)])
def batch_classify(requests: list[ClassifyRequest]):
    """
    Classify multiple emails in a single request.

    Returns a list of classification results in the same order as input.
    """
    classifier = get_classifier()

    results = []
    for req in requests:
        if not req.text or not req.text.strip():
            results.append({"error": "Text cannot be empty"})
            continue

        prediction = classifier.predict(req.text)
        probabilities = classifier.predict_proba(req.text)
        confidence = max(probabilities.values())

        results.append({
            "category": prediction,
            "is_request_for_quote": prediction in RFQ_CATEGORIES,
            "is_listing_service": prediction == 'listing_service',
            "confidence": confidence,
            "probabilities": probabilities
        })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
