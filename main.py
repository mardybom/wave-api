from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from typing import Optional

from canvas_detector import detect_handwritten_letters_from_base64, CanvasInput
from sentence_rearranging import fetch_next_sentence_row
from image_labeling import fetch_random_image_row
from dyslexia_myths import fetch_next_myth_row

app = FastAPI(
    title="Alphabet Mastery API",
    version="2.0.0"
)

@app.post("/alphabet_mastery")
def read_canvas_input(request: CanvasInput):
    """
    Accepts base64 image + expected letter (upper/lowercase), runs OCR, and verifies correctness.
    """
    api_key = os.getenv("GCV_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GCV_API_KEY configuration")

    try:
        result = detect_handwritten_letters_from_base64(
            request.canvas_input,
            api_key,
            request.expected_letter,  # not case sensitive
            request.is_capital,
            request.level
        )
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Sentence Rearranging ---
class SentenceLevelRequest(BaseModel):
    level: str

@app.post("/sentence/next")
def sentence_next(req: SentenceLevelRequest):
    """
    Sentence Rearranging API:
      - Body: { "level": "<value>" }
      - Returns the next unique row for that level (wraps after the last).
    """
    level = (req.level or "").strip()
    if not level:
        raise HTTPException(status_code=400, detail="level is required")

    try:
        row = fetch_next_sentence_row(level)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing DB config env var: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if row is None:
        raise HTTPException(status_code=404, detail=f"No sentence rows found for level '{level}'")

    # Return all columns to the frontend
    return {"status": "success", "data": dict(row)}

# --- Image Labeling ---
class ImageLabelingRequest(BaseModel):
    pass

@app.post("/image_labeling/next")
def get_image_labeling(req: ImageLabelingRequest):
    """
    Fetch a random image row with:
      - image_id
      - base64 image
      - correct label (formatted)
      - 4 fake labels of same length
    """
    try:
        row = fetch_random_image_row()
        if not row:
            raise HTTPException(status_code=404, detail="No rows found in image_labeling table")
        return {"status": "success", "data": row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# If you want strict symmetry in having a request model:
class MythNextRequest(BaseModel):
    pass  # no fields needed; kept to mirror SentenceLevelRequest usage pattern

@app.post("/myth/next")
def myth_next():
    """
    Dyslexia Myths API:
      - Body: {"count": 10}  (optional; defaults to 10)
      - Returns the next N myth/truth rows (wraps after the last).
    """
    try:
        rows = fetch_next_myth_row(batch_size=10)  # new batch function
    except KeyError as e:
        # mirrors your sentence endpoint's config error handling
        raise HTTPException(status_code=500, detail=f"Missing DB config env var: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not rows:
        raise HTTPException(status_code=404, detail="No myth rows found")

    # RealDictCursor returns dict-like rows already; ensure JSON-serializable
    data = [dict(r) for r in rows]

    return {
        "status": "success",
        "count": len(data),
        "data": data
    }