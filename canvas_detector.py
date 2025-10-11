"""
Canvas Handwriting Detection Module

This module provides functionality to detect handwritten letters
from base64-encoded canvas images using the Google Vision API.
It validates input parameters, ensures consistency with difficulty
levels and letter casing, and returns match accuracy and mismatch details.
"""


from collections import defaultdict
from typing import List, Dict
import base64
import requests
from fastapi import HTTPException
from pydantic import BaseModel


class CanvasInput(BaseModel):
    """Model for handwritten letter detection input."""
    canvas_input: str          # User-entered handwritten alphabet in base64 (Full HD)
    expected_letter: str       # Hardcoded expected alphabet from the frontend (e.g., "hb", "A", "xy")
    is_capital: str            # "capital" or "small"
    level: str                 # "easy" (1 letter) or "hard" (2 letters)


def detect_handwritten_letters_from_base64(
    b64_image: str,
    api_key: str,
    expected_letter: str,
    is_capital: str,
    level: str
):
    """Detect handwritten letters using Google Vision API."""
    try:
        # Validate expected_letter case-insensitively
        if expected_letter is None:
            raise HTTPException(status_code=400, detail="expected_letter is required")

        expected_letter = expected_letter.strip()
        if not expected_letter.isalpha():
            raise HTTPException(status_code=400, detail="expected_letter must be Aa-Zz")

        if (
            (is_capital == "capital" and expected_letter.islower())
            or (is_capital == "small" and expected_letter.isupper())
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"is_capital ({is_capital}) does not match "
                    f"with expected_letter ({expected_letter})"
                ),
            )

        if (
            (level == "easy" and len(expected_letter) > 1)
            or (level == "hard" and len(expected_letter) == 1)
        ):
            level_description = "2 letters" if level == "hard" else "1 letter"
            raise HTTPException(
                status_code=400,
                detail=(
                    f"expected_letter '{expected_letter}' in {level} "
                    f"level must be {level_description}"
                ),
            )

        # Support data URLs and raw base64
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        b64_image = b64_image.strip().replace("\n", "").replace("\r", "")

        if not b64_image:
            raise HTTPException(status_code=400, detail="Empty base64 image")

        # Validate base64
        try:
            base64.b64decode(b64_image, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

        def call_vision(feature_type: str):
            """Helper to call the Google Vision API."""
            payload = {
                "requests": [
                    {
                        "image": {"content": b64_image},
                        "features": [{"type": feature_type}],
                        "imageContext": {"languageHints": ["en"]},
                    }
                ]
            }
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("responses", [{}])[0]
            if "error" in data:
                raise HTTPException(
                    status_code=502,
                    detail=data["error"].get("message", "Vision API error"),
                )
            return data

        try:
            # First pass: TEXT_DETECTION (faster, better for single letters)
            res = call_vision("TEXT_DETECTION")

            # If no text found, fallback to DOCUMENT_TEXT_DETECTION
            if not res.get("fullTextAnnotation") and not res.get("textAnnotations"):
                res = call_vision("DOCUMENT_TEXT_DETECTION")

        except requests.exceptions.RequestException as req_error:
            raise HTTPException(
                status_code=502,
                detail=f"Vision API request failed: {req_error}",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected Vision API error: {exc}",
            )

        doc = res.get("fullTextAnnotation")
        letters: List[Dict] = []

        if doc:
            for page in doc.get("pages", []):
                for block in page.get("blocks", []):
                    for para in block.get("paragraphs", []):
                        for word in para.get("words", []):
                            for symbol in word.get("symbols", []):
                                ch = symbol.get("text", "")
                                if (ch.isalpha() or ch == "@") and ord(ch) < 128:
                                    if ch.lower() in "cxvusmwyzp":
                                        ch = ch.upper() if is_capital == "capital" else ch.lower()
                                    letters.append({"letter": ch})

        # Case-insensitive verification
        expected_up = expected_letter
        total_alpha = len(letters)
        matches = [x for x in letters if x["letter"] in expected_up]
        match_count = len(matches)

        mismatch_counts = defaultdict(int)
        mismatch_top_conf = defaultdict(float)

        for x in letters:
            up = x["letter"]
            if up not in expected_up:
                mismatch_counts[up] += 1

        for y in expected_letter:
            if y not in [x["letter"] for x in letters]:
                mismatch_counts[y] += 1

        mismatches = [
            {"letter": k, "count": v}
            for k, v in sorted(
                mismatch_counts.items(),
                key=lambda kv: (-kv[1]),
            )
        ]

        ratio = (
            (len(expected_letter) - len(mismatches)) / len(expected_letter)
            if expected_letter
            else 0.0
        )
        is_correct = len(mismatches) == 0

        # Generate reason message
        if not is_correct:
            if len(mismatches) > 0:
                reason = f"You have {len(mismatches)} mismatched alphabets"
            elif ratio < 0.60:
                reason = "You're partially correct."
            else:
                reason = "Please try again."
        else:
            reason = "Match found"

        return {
            "expected_letter": expected_up,
            "is_correct": is_correct,
            "reason": reason,
            "detected_count": total_alpha,
            "match_count": match_count,
            "match_ratio": round(ratio, 3),
            "letters": letters,
            "mismatches": mismatches,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")