from pydantic import BaseModel
from fastapi import HTTPException
import base64, requests

class CanvasInput(BaseModel):
    canvas_input: str  # base64 or data URL

def detect_handwritten_letters_from_base64(b64_image: str, api_key: str):
    try:
        # Accept data URLs and raw base64
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]

        # Optional: clean accidental whitespace/newlines
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
        letters = []
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
        return letters

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")