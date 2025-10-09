"""
Canvas Detector
Performs OCR-based verification of handwritten letters using Google Cloud Vision (REST API).
Improved for single or dual handwritten letters.
"""

import base64
import requests
import statistics
from typing import List, Dict
from fastapi import HTTPException
from pydantic import BaseModel
import cv2, numpy as np, io
from PIL import Image

# ---------- Request model ----------
class CanvasInput(BaseModel):
    canvas_input: str           # Base64 image input (from frontend)
    expected_letter: str        # Hardcoded expected letter(s), e.g., "A" or "ab"
    is_capital: str             # "capital" or "small"
    level: str                  # "easy" (1 letter) or "hard" (2 letters)

# ---------- Preprocessing ----------
def _preprocess_base64_for_ocr(b64_image: str) -> str:
    """Preprocess base64 image to improve OCR accuracy for handwritten letters."""
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_image)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    g = np.array(img)

    # Binarize + invert so black strokes on white background
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:
        bw = 255 - bw

    # Morph close (connects gaps in i/j, W/M)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    # Keep largest connected component (ignore noise)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, _ = stats[idx]
        roi = bw[y:y+h, x:x+w]
    else:
        roi = bw

    # Square-pad and center
    size = max(roi.shape[:2])
    canvas = np.zeros((size, size), dtype=np.uint8)
    oy, ox = (size - roi.shape[0]) // 2, (size - roi.shape[1]) // 2
    canvas[oy:oy+roi.shape[0], ox:ox+roi.shape[1]] = roi

    # Slight dilation for faint lines
    canvas = cv2.dilate(canvas, np.ones((2,2), np.uint8), iterations=1)

    # Resize to 128x128 for consistency
    canvas = cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_AREA)

    # Invert back for Vision API (black text on white)
    canvas = 255 - canvas

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to preprocess image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------- Letter detection ----------
def detect_handwritten_letters_from_base64(b64_image: str, api_key: str, expected_letter: str, is_capital: str, level: str):
    """
    Detects handwritten letters and verifies if they match the expected letter(s).
    """

    # ---- Validate input ----
    if not expected_letter or not expected_letter.strip():
        raise HTTPException(status_code=400, detail="expected_letter is required")
    if is_capital not in ("capital", "small"):
        raise HTTPException(status_code=400, detail="is_capital must be 'capital' or 'small'")
    if level not in ("easy", "hard"):
        raise HTTPException(status_code=400, detail="level must be 'easy' or 'hard'")

    expected_letter = expected_letter.strip()
    if not expected_letter.isalpha():
        raise HTTPException(status_code=400, detail="expected_letter must contain only letters")

    if level == "easy" and len(expected_letter) != 1:
        raise HTTPException(status_code=400, detail="easy level must have exactly 1 expected letter")
    if level == "hard" and len(expected_letter) != 2:
        raise HTTPException(status_code=400, detail="hard level must have exactly 2 expected letters")

    expected_norm = expected_letter.upper() if is_capital == "capital" else expected_letter.lower()

    # ---- Preprocess image ----
    try:
        processed_b64 = _preprocess_base64_for_ocr(b64_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    # ---- Call GCV ----
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": processed_b64},
            "features": [{"type": "TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["en"]},
        }]
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Vision API request failed: {e}")

    res = resp.json().get("responses", [{}])[0]
    if "error" in res:
        raise HTTPException(status_code=502, detail=res["error"].get("message", "Vision API error"))

    # ---- Extract letters ----
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
                            if ch.isalpha():
                                ch = ch.upper() if is_capital == "capital" else ch.lower()
                                letters.append({"letter": ch, "confidence": round(conf, 3)})

    if not letters:
        raise HTTPException(status_code=422, detail="No letters detected. Try writing more clearly.")

    total_letters = len(letters)
    matches = [x for x in letters if x["letter"] in expected_norm]
    mismatch = [x for x in letters if x["letter"] not in expected_norm]
    match_count = len(matches)
    top_match_conf = max([m["confidence"] for m in matches], default=0.0)
    ratio = (match_count / total_letters) if total_letters else 0

    # ---- Determine correctness ----
    if level == "easy":
        is_correct = top_match_conf >= 0.65 and ratio >= 0.5
    else:
        is_correct = len(set(expected_norm).intersection([x["letter"] for x in matches])) == len(expected_norm) \
                     and top_match_conf >= 0.55

    reason = (
        "Match found with sufficient confidence."
        if is_correct else
        ("Low confidence or partial mismatch." if matches else "No matching letter found.")
    )

    return {
        "expected_letter": expected_norm,
        "is_correct": is_correct,
        "reason": reason,
        "detected_count": total_letters,
        "match_count": match_count,
        "top_match_confidence": round(top_match_conf, 3),
        "match_ratio": round(ratio, 3),
        "letters": letters,
        "mismatches": mismatch,
    }