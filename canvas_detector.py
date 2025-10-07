"""
Canvas Detector
Handles OCR-based verification of handwritten letters using Google Cloud Vision.
"""

import base64
import requests
import statistics
from collections import defaultdict
from typing import List, Dict
from fastapi import HTTPException
from pydantic import BaseModel


class CanvasInput(BaseModel):
    canvas_input: str
    expected_letter: str
    is_capital: str
    level: str


MIN_CONF = 0.30
MIN_RATIO = 0.60


def detect_handwritten_letters_from_base64(
    b64_image: str,
    api_key: str,
    expected_letter: str,
    is_capital: str,
    level: str,
) -> dict:
    """Detect and verify handwritten letters using Google Cloud Vision."""

    # ---------------- Validate Inputs ---------------- #
    if not expected_letter or not expected_letter.strip():
        raise HTTPException(status_code=400, detail="expected_letter is required")
    expected_letter = expected_letter.strip()

    if not expected_letter.isalpha():
        raise HTTPException(status_code=400, detail="expected_letter must be a letter (A-Z or a-z)")

    if (is_capital == "capital" and expected_letter.islower()) or (
        is_capital == "small" and expected_letter.isupper()
    ):
        raise HTTPException(
            status_code=400,
            detail=f"is_capital ({is_capital}) does not match expected_letter ({expected_letter})",
        )

    if (level == "easy" and len(expected_letter) > 1) or (level == "hard" and len(expected_letter) == 1):
        msg = "2 letters" if level == "hard" else "1 letter"
        raise HTTPException(
            status_code=400,
            detail=f"expected_letter '{expected_letter}' in {level} level must be {msg}",
        )

    # ---------------- Base64 Handling ---------------- #
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    b64_image = "".join(b64_image.split())
    if not b64_image:
        raise HTTPException(status_code=400, detail="Empty base64 image")

    try:
        base64.b64decode(b64_image, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # ---------------- Google Vision API ---------------- #
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": b64_image},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["en"]},
        }]
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as re:
        raise HTTPException(status_code=502, detail=f"Vision API request failed: {re}")

    res = resp.json().get("responses", [{}])[0]
    if "error" in res:
        raise HTTPException(status_code=502, detail=res["error"].get("message", "Vision API error"))

    doc = res.get("fullTextAnnotation")
    letters: List[Dict] = []

    if doc:
        for page in doc.get("pages", []):
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        for symbol in word.get("symbols", []):
                            ch = symbol.get("text", "")
                            conf = float(symbol.get("confidence", 0.0) or 0.0)
                            if (ch.isalpha() or ch == "@") and ord(ch) < 128:
                                ch = ch.upper() if is_capital == "capital" else ch.lower()
                                letters.append({"letter": ch, "confidence": round(conf, 3)})

    total_alpha = len(letters)
    matches = [x for x in letters if x["letter"] in expected_letter]
    match_count = len(matches)
    top_match_conf = statistics.mean([m["confidence"] for m in matches]) if matches else 0.0

    mismatch_counts = defaultdict(int)
    mismatch_top_conf = defaultdict(float)

    for x in letters:
        if x["letter"] not in expected_letter:
            mismatch_counts[x["letter"]] += 1
            mismatch_top_conf[x["letter"]] = max(mismatch_top_conf[x["letter"]], x["confidence"])

    for y in expected_letter:
        if y not in [x["letter"] for x in letters]:
            mismatch_counts[y] += 1
            mismatch_top_conf[y] = 0

    mismatches = [
        {"letter": k, "count": v, "top_confidence": round(mismatch_top_conf[k], 3)}
        for k, v in sorted(mismatch_counts.items(), key=lambda kv: (-kv[1], -mismatch_top_conf[kv[0]]))
    ]

    ratio = ((len(expected_letter) - len(mismatches)) / len(expected_letter)) if expected_letter else 0.0
    is_correct = len(mismatches) == 0 and (top_match_conf >= MIN_CONF and ratio >= MIN_RATIO)

    # ---------------- Feedback ---------------- #
    if is_correct:
        reason = "Match found with sufficient confidence."
    elif mismatches:
        reason = f"You have {len(mismatches)} mismatched alphabet(s)."
    elif top_match_conf < MIN_CONF:
        reason = f"Low confidence ({top_match_conf:.2f})"
    elif ratio < MIN_RATIO:
        reason = "You're partially correct."
    else:
        reason = "Please try again."

    return {
        "expected_letter": expected_letter,
        "is_correct": is_correct,
        "reason": reason,
        "detected_count": total_alpha,
        "match_count": match_count,
        "top_match_confidence": round(top_match_conf, 3),
        "match_ratio": round(ratio, 3),
        "letters": letters,
        "mismatches": mismatches,
    }
