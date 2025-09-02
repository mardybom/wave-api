from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from canvas_detector import detect_handwritten_letters_from_base64, CanvasInput
import os

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

@app.get("/")
def read_root():
    return {"Hello": "World"}

# ---------- Response models----------
class LetterOut(BaseModel):
    letter: str
    confidence: float

class MismatchOut(BaseModel):
    letter: str
    count: int
    top_confidence: float

class VerificationResponse(BaseModel):
    status: Literal["success"]
    expected_letter: str
    is_correct: bool
    reason: str
    detected_count: int
    match_count: int
    top_match_confidence: float
    match_ratio: float
    letters: List[LetterOut]
    mismatches: List[MismatchOut]

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
            request.expected_letter  # not case sensitive
        )
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")