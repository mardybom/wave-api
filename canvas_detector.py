"""
Canvas Detector
Handles OCR-based verification of handwritten letters using Google Cloud Vision.
"""

import base64
import requests
from typing import List, Dict
from fastapi import HTTPException
from pydantic import BaseModel
import cv2, numpy as np, io
from PIL import Image

# ---------- Request model ----------
class CanvasInput(BaseModel):
    canvas_input: str           # base64 or data URL
    expected_letter: str        # "a" for easy, "ab" for hard
    is_capital: str             # "capital" | "small"
    level: str                  # "easy" | "hard"

def _preprocess_base64_for_ocr(b64_image: str) -> str:
    # strip header if present
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    # decode
    img_bytes = base64.b64decode("".join(b64_image.split()))
    img = Image.open(io.BytesIO(img_bytes)).convert("L")     # grayscale
    g = np.array(img)

    # binarize (Otsu), make strokes dark on light bg
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure black background / white strokes for processing
    # (we want stroke=white for connected components below)
    if bw.mean() < 127:
        bw = 255 - bw

    # connect small gaps (helps i/j dots, W/M intersections)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    # largest connected component (ignore background)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, _ = stats[idx]
        roi = bw[y:y+h, x:x+w]
    else:
        roi = bw

    # square pad + center
    size = max(roi.shape[:2])
    canvas = np.zeros((size, size), dtype=np.uint8)
    oy = (size - roi.shape[0]) // 2
    ox = (size - roi.shape[1]) // 2
    canvas[oy:oy+roi.shape[0], ox:ox+roi.shape[1]] = roi

    # optional: thicken faint strokes a touch
    canvas = cv2.dilate(canvas, np.ones((2,2), np.uint8), iterations=1)

    # resize to 128x128 for consistent OCR
    canvas = cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_AREA)

    # invert back to black stroke on white bg (Vision prefers it)
    canvas = 255 - canvas

    # re-encode as PNG base64
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise HTTPException(status_code=500, detail="Preprocessing failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")

# Tunables
TOP_CONF_EASY = 0.70   # strong single-letter evidence
MIN_RATIO = 0.60       # dominance threshold across all detected letters
MIN_CONF_SYMBOL = 0.30 # ignore ultra-weak symbols for "primary" choice

def detect_handwritten_letters_from_base64(
    b64_image: str,
    api_key: str,
    expected_letter: str,
    is_capital: str,
    level: str,
) -> dict:
    """
    Verifies that the canvas image contains the expected letter(s).
    Modes:
      - capital-easy  : 1 letter (uppercased)
      - capital-hard  : 2 letters (uppercased), order required
      - small-easy    : 1 letter (lowercased)
      - small-hard    : 2 letters (lowercased), order required
    """

    # ---------- Validate & normalize inputs ----------
    if not expected_letter or not expected_letter.strip():
        raise HTTPException(status_code=400, detail="expected_letter is required")
    if is_capital not in ("capital", "small"):
        raise HTTPException(status_code=400, detail="is_capital must be 'capital' or 'small'")
    if level not in ("easy", "hard"):
        raise HTTPException(status_code=400, detail="level must be 'easy' or 'hard'")

    expected_letter = expected_letter.strip()
    if not expected_letter.isalpha():
        raise HTTPException(status_code=400, detail="expected_letter must be letters only")

    if level == "easy" and len(expected_letter) != 1:
        raise HTTPException(status_code=400, detail="For 'easy', expected_letter must be exactly 1 letter")
    if level == "hard" and len(expected_letter) != 2:
        raise HTTPException(status_code=400, detail="For 'hard', expected_letter must be exactly 2 letters, e.g. 'ab'")

    # Normalize case according to is_capital (so the comparison is consistent)
    expected_norm = expected_letter.upper() if is_capital == "capital" else expected_letter.lower()

    # ---------- Base64 cleanup ----------
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    b64_image = "".join(b64_image.split())
    if not b64_image:
        raise HTTPException(status_code=400, detail="Empty base64 image")
    try:
        base64.b64decode(b64_image, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # ---------- Call GCV ----------
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    # before calling GCV:
    processed_b64 = _preprocess_base64_for_ocr(b64_image)

    payload = {
        "requests": [{
            "image": {"content": processed_b64},
            "features": [{"type": "TEXT_DETECTION"}],  # was DOCUMENT_TEXT_DETECTION
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

    # ---------- Extract letters & sequence ----------
    doc = res.get("fullTextAnnotation")
    letters: List[Dict] = []
    seq_chars: List[str] = []

    if doc:
        for page in doc.get("pages", []):
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        for symbol in word.get("symbols", []):
                            ch = symbol.get("text", "")
                            conf = float(symbol.get("confidence", 0.0) or 0.0)
                            if ch.isalpha() and ord(ch) < 128:
                                ch = ch.upper() if is_capital == "capital" else ch.lower()
                                letters.append({"letter": ch, "confidence": round(conf, 3)})
                                if conf >= MIN_CONF_SYMBOL:
                                    seq_chars.append(ch)

    total_alpha = len(letters)
    seq = "".join(seq_chars)

    # Early exit if nothing readable
    if total_alpha == 0:
        return {
            "mode": f"{is_capital}-{level}",
            "expected_letter": expected_norm,
            "is_correct": False,
            "reason": "No letters detected by OCR.",
            "detected_count": 0,
            "match_count": 0,
            "top_match_confidence": 0.0,
            "match_ratio": 0.0,
            "sequence": seq,
            "sequence_match": False,
            "letters": letters,
            "mismatches": [],
        }

    # Build quick stats per letter
    from collections import defaultdict
    count = defaultdict(int)
    top_conf = defaultdict(float)
    for x in letters:
        l = x["letter"]
        c = x["confidence"]
        count[l] += 1
        top_conf[l] = max(top_conf[l], c)
    
    # ---------- EASY: one expected letter ----------
    if level == "easy":
        exp = expected_norm
        allowed = EQUIV.get(exp, {exp})

        matches_list = [x for x in letters if x["letter"] in allowed]
        match_count = len(matches_list)
        ratio = (match_count / total_alpha) if total_alpha else 0.0

        # top confidence among matches (incl. equivalents)
        top = max((m["confidence"] for m in matches_list), default=0.0)

        # Primary symbol based on confidence across all letters
        primary_letter = max(top_conf, key=lambda k: top_conf[k])
        primary_conf = top_conf[primary_letter]

        # Accept exact primary with normal threshold; accept equivalent with stricter threshold (+0.1)
        primary_ok = (
            (primary_letter == exp and primary_conf >= TOP_CONF_EASY) or
            (primary_letter in allowed and primary_letter != exp and primary_conf >= (TOP_CONF_EASY + 0.10))
        )

        is_correct = primary_ok or (ratio >= MIN_RATIO and match_count >= 1)

        # mismatches: everything not expected (exact), plus expected if missing
        mismatches = []
        for k, v in count.items():
            if k not in allowed:
                mismatches.append({"letter": k, "count": v, "top_confidence": round(top_conf[k], 3)})
        if match_count == 0:
            mismatches.append({"letter": exp, "count": 1, "top_confidence": 0.0})
        mismatches.sort(key=lambda kv: (-kv["count"], -kv["top_confidence"]))

        reason = (
            "Primary letter (or allowed equivalent) matches with strong confidence."
            if primary_ok else
            ("Expected letter (or equivalent) dominates detections."
            if (ratio >= MIN_RATIO and match_count >= 1) else
            ("No matching letter detected." if match_count == 0
            else (f"Low confidence on primary ({primary_conf:.2f})." if primary_letter in allowed
                    else "Expected letter not primary.")))
        )

        return {
            "mode": f"{is_capital}-easy",
            "expected_letter": exp,
            "is_correct": is_correct,
            "reason": reason,
            "detected_count": total_alpha,
            "match_count": match_count,
            "top_match_confidence": round(top, 3),
            "match_ratio": round(ratio, 3),
            "sequence": seq,
            "sequence_match": (exp in seq),
            "letters": letters,
            "mismatches": mismatches,
        }


    # ---------- HARD: two expected letters (order required) ----------
    exp1, exp2 = expected_norm[0], expected_norm[1]
    c1, c2 = count[exp1], count[exp2]
    top1, top2 = top_conf.get(exp1, 0.0), top_conf.get(exp2, 0.0)
    total_matches = c1 + c2
    ratio = (total_matches / total_alpha) if total_alpha else 0.0

    # Require order "exp1exp2" somewhere in the detected sequence.
    # To relax order, change next line to: sequence_ok = (c1 >= 1 and c2 >= 1)
    sequence_ok = (exp1 + exp2 in seq)

    both_present = (c1 >= 1 and c2 >= 1)
    conf_ok = (min(top1, top2) >= TOP_CONF_EASY)
    ratio_ok = (ratio >= MIN_RATIO)

    is_correct = both_present and sequence_ok and (conf_ok or ratio_ok)

    # mismatches: letters not in {exp1, exp2} + missing expected ones
    mismatches = []
    for k, v in count.items():
        if k not in (exp1, exp2):
            mismatches.append({"letter": k, "count": v, "top_confidence": round(top_conf[k], 3)})
    if c1 == 0:
        mismatches.append({"letter": exp1, "count": 1, "top_confidence": 0.0})
    if c2 == 0:
        mismatches.append({"letter": exp2, "count": 1, "top_confidence": 0.0})
    mismatches.sort(key=lambda kv: (-kv["count"], -kv["top_confidence"]))

    if not both_present:
        reason = "Both letters must be present at least once."
    elif not sequence_ok:
        reason = f"Letters not detected in the required order '{exp1}{exp2}'."
    elif conf_ok:
        reason = "Both letters detected with strong confidence."
    elif ratio_ok:
        reason = "Expected pair dominates the detections."
    else:
        reason = "Low confidence and low ratio for the expected pair."

    return {
        "mode": f"{is_capital}-hard",
        "expected_letter": expected_norm,
        "is_correct": is_correct,
        "reason": reason,
        "detected_count": total_alpha,
        "match_count": total_matches,
        "top_match_confidence": round(min(top1, top2), 3),
        "top_match_confidence_per_letter": {exp1: round(top1, 3), exp2: round(top2, 3)},
        "match_ratio": round(ratio, 3),
        "sequence": seq,
        "sequence_match": sequence_ok,
        "letters": letters,
        "mismatches": mismatches,
    }

EQUIV = {
    # rounded vs open
    'o': {'o','O','0','c','C','q','Q'}, 'O': {'o','O','0','C','Q'},
    'c': {'c','C','o','O'},
    # V/U/Y family
    'v': {'v','V','u','U','y','Y'}, 'V': {'v','V','U','Y'},
    'u': {'u','U','v','V'}, 'U': {'u','U','V'},
    'y': {'y','Y','v','V'},
    # W/M (zigzag peaks)
    'w': {'w','W','vv','VV'}, 'W': {'w','W','VV'},
    'm': {'m','M','nn'}, 'M': {'m','M','NN'},
    # verticals
    'l': {'l','I','1','|'}, 'I': {'I','l','1','|'},
    # loops / stems
    'p': {'p','P','b'}, 'P': {'P','p','B'},
    'q': {'q','Q','g'}, 'Q': {'Q','q'},
    # crosses / angles
    'x': {'x','X','k','K'}, 'X': {'x','X'},
    'z': {'z','Z','2'},     'Z': {'z','Z'},
}
