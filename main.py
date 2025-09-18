from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel

from canvas_detector import detect_handwritten_letters_from_base64, CanvasInput
from db import fetch_row_by_level


app = FastAPI(
    title="Alphabet Mastery API",
    version="1.0.0",
    description="Accepts a base64 canvas image and an expected letter (A–Z), OCRs it, and verifies correctness."
)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
allow_credentials = False if ALLOWED_ORIGINS == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
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
      - No schema creation; expects DB tables already exist.
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

    # Return the full row (all columns) to the frontend
    return {"status": "success", "data": dict(row)}