"""
Canvas Detector
Handles OCR-based verification of handwritten letters using Google Cloud Vision.
"""

import base64
import requests
from typing import List, Dict
from fastapi import HTTPException
from pydantic import BaseModel

# ---------- Request model ----------
class CanvasInput(BaseModel):
    canvas_input: str           # base64 or data URL
    expected_letter: str        # "a" for easy, "ab" for hard
    is_capital: str             # "capital" | "small"
    level: str                  # "easy" | "hard"

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
        matches = count[exp]
        match_count = matches
        ratio = (matches / total_alpha) if total_alpha else 0.0
        top = top_conf.get(exp, 0.0)

        # Decide: match the expected letter (primary or dominant)
        # 1) Primary check: the highest-confidence symbol equals expected AND has strong confidence
        # 2) Dominance check: expected letter is the majority of detections
        # Primary symbol = max top_conf across letters
        primary_letter = max(top_conf, key=lambda k: top_conf[k])
        primary_conf = top_conf[primary_letter]

        is_correct = (
            (primary_letter == exp and primary_conf >= TOP_CONF_EASY) or
            (ratio >= MIN_RATIO and matches >= 1)
        )

        # Build mismatches list (everything not expected, plus expected if missing)
        mismatches = []
        for k, v in count.items():
            if k != exp:
                mismatches.append({"letter": k, "count": v, "top_confidence": round(top_conf[k], 3)})
        if matches == 0:
            mismatches.append({"letter": exp, "count": 1, "top_confidence": 0.0})
        mismatches.sort(key=lambda kv: (-kv["count"], -kv["top_confidence"]))

        reason = (
            "Primary letter matches expected with strong confidence."
            if (primary_letter == exp and primary_conf >= TOP_CONF_EASY) else
            ("Expected letter dominates detections."
             if (ratio >= MIN_RATIO and matches >= 1) else
             ("No matching letter detected." if matches == 0
              else (f"Low confidence on primary ({primary_conf:.2f})." if primary_letter == exp else "Expected letter not primary.")))
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