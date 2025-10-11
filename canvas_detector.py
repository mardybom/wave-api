from pydantic import BaseModel
from fastapi import HTTPException
import base64, requests, statistics
from typing import List, Dict
from collections import defaultdict

class CanvasInput(BaseModel):
    canvas_input: str           # user entered handwritten alphabet in base64 full hd.
    expected_letter: str        # Hardcoded expected alphabet from the frontend(eg: "hb", "A", "xy")
    is_capital: str             # "capital" or "small"
    level: str                  # "easy" (1 letter) or "hard" (2 letters)

def detect_handwritten_letters_from_base64(b64_image: str, api_key: str, expected_letter: str, is_capital: str, level: str):
    try:
        # validate expected_letter case-insensitively
        if expected_letter is None:
            raise HTTPException(status_code=400, detail="expected_letter is required")
        expected_letter = expected_letter.strip()
        if not expected_letter.isalpha():
            raise HTTPException(status_code=400, detail=f"expected_letter must be Aa-Zz")
        
        if (is_capital == "capital" and expected_letter.islower()) or (is_capital == "small" and expected_letter.isupper()):
            raise HTTPException(status_code=400, detail=f"is_capital ({is_capital}) does not match with expected_letter ({expected_letter})")
        
        if (level == "easy" and len(expected_letter) > 1) or (level == "hard" and len(expected_letter) == 1):
            level_description = "2 letters" if level == "hard" else "1 letter"
            raise HTTPException(status_code=400, detail= f"expected_letter '{expected_letter}' in {level} level must be {level_description}")

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
            payload = {
                "requests": [{
                    "image": {"content": b64_image},
                    "features": [{"type": feature_type}],
                    "imageContext": {"languageHints": ["en"]},
                }]
            }
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("responses", [{}])[0]
            if "error" in data:
                raise HTTPException(status_code=502, detail=data["error"].get("message", "Vision API error"))
            return data

        try:
            # First pass: TEXT_DETECTION (faster, better for single letters)
            res = call_vision("TEXT_DETECTION")

            # If no text found, fallback to DOCUMENT_TEXT_DETECTION
            if not res.get("fullTextAnnotation") and not res.get("textAnnotations"):
                res = call_vision("DOCUMENT_TEXT_DETECTION")

        except requests.exceptions.RequestException as re:
            raise HTTPException(status_code=502, detail=f"Vision API request failed: {re}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected Vision API error: {e}")

        doc = res.get("fullTextAnnotation")
        letters: List[Dict] = []
        if doc:
            for page in doc.get("pages", []):
                for block in page.get("blocks", []):
                    for para in block.get("paragraphs", []):
                        for word in para.get("words", []):
                            for symbol in word.get("symbols", []):
                                ch = symbol.get("text", "")
                                # conf = float(symbol.get("confidence", 0.0) or 0.0)
                                if (ch.isalpha() or ch == '@') and ord(ch) < 128:
                                    if ch.lower() in 'cxvusmwyzp':
                                        if is_capital == "capital":
                                            ch = ch.upper()
                                        else:
                                            ch = ch.lower()
                                    letters.append({"letter": ch})
                                    # letters.append({"letter": ch, "confidence": round(conf, 3)})

        # Case-insensitive verification
        # expected_up = expected_letter.upper()
        expected_up = expected_letter
        total_alpha = len(letters)
        # matches = [x for x in letters if x["letter"].upper() == expected_up]
        matches = [x for x in letters if x["letter"] in expected_up]
        match_count = len(matches)
        # top_match_conf = max([m["confidence"] for m in matches], default=0.0)
        # top_match_conf = statistics.mean([m["confidence"] for m in matches]) if len(matches) > 0 else 0
        # ratio = (match_count / total_alpha) if total_alpha else 0.0
        # is_correct = (match_count == len(expected_up)) and (mismatch_counts == 0) and (top_match_conf >= 0.70 or ratio >= 0.60)

        mismatch_counts = defaultdict(int)
        mismatch_top_conf = defaultdict(float)
        for x in letters:
            # up = x["letter"].upper()
            up = x["letter"]
            if up not in expected_up:
                mismatch_counts[up] += 1
                # mismatch_top_conf[up] = max(mismatch_top_conf[up], x["confidence"])

        for y in expected_letter:
            if y not in [x["letter"] for x in letters]:
                mismatch_counts[y] += 1
                # mismatch_top_conf[y] = 0

        # mismatches = [
        #     {"letter": k, "count": v, "top_confidence": round(mismatch_top_conf[k], 3)}
        #     for k, v in sorted(mismatch_counts.items(), key=lambda kv: (-kv[1], -mismatch_top_conf[kv[0]]))
        # ]
        mismatches = [
            {"letter": k, "count": v}
            for k, v in sorted(mismatch_counts.items(), key=lambda kv: (-kv[1]))
        ]
        ratio = ((len(expected_letter) - len(mismatches)) / len(expected_letter)) if expected_letter else 0.0
        is_correct = (len(mismatches)==0) 
        # and (top_match_conf >= 0.30 and ratio >= 0.60)
        
        reason = ""
        if not is_correct:
            if (len(mismatches) > 0): # when user writes extra letters, or they are missing some letters
                reason = f"You have {len(mismatches)} mismatched alphabets"
            # elif (top_match_conf < 0.30):
            #     reason = f"Low confidence ({top_match_conf:.2f})"
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
            # "top_match_confidence": round(top_match_conf, 3),
            "match_ratio": round(ratio, 3),
            "letters": letters,
            "mismatches": mismatches
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")