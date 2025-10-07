"""
GCV Configuration Module
Centralizes fetching and validation of the Google Cloud Vision API key.
"""

import os
import logging
from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")


def get_gcv_api_key() -> str:
    """
    Retrieve the Google Cloud Vision API key from environment variables.

    Returns:
        str: The API key.

    Raises:
        HTTPException: If the API key is missing.
    """
    api_key = os.getenv("GCV_API_KEY")
    if not api_key:
        logger.error("Missing GCV_API_KEY environment variable.")
        raise HTTPException(status_code=500, detail="Missing GCV_API_KEY configuration")
    return api_key
