from pydantic import BaseModel
from fastapi import HTTPException
import base64, requests
from typing import List, Dict
from collections import defaultdict

class CanvasInput(BaseModel):
    canvas_input: str           # base64
    expected_letter: str        # validate manually

def detect_handwritten_letters_from_base64(b64_image: str, api_key: str, expected_letter: str):
    try:
        # --- validate expected_letter case-insensitively ---
        if expected_letter is None:
            raise HTTPException(status_code=400, detail="expected_letter is required")
        expected_letter = expected_letter.strip()
        if len(expected_letter) != 1 or not expected_letter.isalpha():
            raise HTTPException(status_code=400, detail="expected_letter must be a single A–Z letter")

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
        payload = {
            "requests": [{
                "image": {"content": b64_image},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["en"]}
            }]
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
        except requests.exceptions.RequestException as re:
            raise HTTPException(status_code=502, detail=f"Vision API request failed: {re}")

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Vision API HTTP {resp.status_code}: {resp.text}")

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
                                if ch.isalpha() and ord(ch) < 128:
                                    letters.append({"letter": ch, "confidence": round(conf, 3)})

        # Case-insensitive verification
        expected_up = expected_letter.upper()
        total_alpha = len(letters)
        matches = [x for x in letters if x["letter"].upper() == expected_up]
        match_count = len(matches)
        top_match_conf = max([m["confidence"] for m in matches], default=0.0)
        ratio = (match_count / total_alpha) if total_alpha else 0.0
        is_correct = (match_count >= 1) and (top_match_conf >= 0.70 or ratio >= 0.60)

        mismatch_counts = defaultdict(int)
        mismatch_top_conf = defaultdict(float)
        for x in letters:
            up = x["letter"].upper()
            if up != expected_up:
                mismatch_counts[up] += 1
                mismatch_top_conf[up] = max(mismatch_top_conf[up], x["confidence"])

        mismatches = [
            {"letter": k, "count": v, "top_confidence": round(mismatch_top_conf[k], 3)}
            for k, v in sorted(mismatch_counts.items(), key=lambda kv: (-kv[1], -mismatch_top_conf[kv[0]]))
        ]

        return {
            "expected_letter": expected_up,
            "is_correct": is_correct,
            "reason": (
                "match found with sufficient confidence" if is_correct
                else ("no matching letter detected" if match_count == 0
                      else f"low confidence ({top_match_conf:.2f}) and low ratio ({ratio:.2f})")
            ),
            "detected_count": total_alpha,
            "match_count": match_count,
            "top_match_confidence": round(top_match_conf, 3),
            "match_ratio": round(ratio, 3),
            "letters": letters,
            "mismatches": mismatches
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
