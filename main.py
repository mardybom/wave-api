from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from gcv_config import get_gcv_api_key
from canvas_detector import detect_handwritten_letters_from_base64, CanvasInput
from sentence_rearranging import fetch_next_sentence_row
from image_labeling import fetch_random_image_row
from dyslexia_myths import fetch_next_myth_row
from chatbot import get_parent_answer

app = FastAPI(title="Alphabet Mastery API", version="3.0.0")


# ---------------- Alphabet Mastery ---------------- #
@app.post("/alphabet_mastery")
def read_canvas_input(request: CanvasInput):
    api_key = get_gcv_api_key()
    try:
        result = detect_handwritten_letters_from_base64(
            request.canvas_input,
            api_key,
            request.expected_letter,
            request.is_capital,
            request.level,
        )
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ---------------- Sentence Rearranging ---------------- #
class SentenceLevelRequest(BaseModel):
    level: str


@app.post("/sentence/next")
def sentence_next(req: SentenceLevelRequest):
    level = (req.level or "").strip()
    if not level:
        raise HTTPException(status_code=400, detail="level is required")

    try:
        row = fetch_next_sentence_row(level)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No rows found for level '{level}'")
        return {"status": "success", "data": dict(row)}
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing DB config: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ---------------- Image Labeling ---------------- #
@app.post("/image_labeling/next")
def get_image_labeling():
    try:
        row = fetch_random_image_row()
        if not row:
            raise HTTPException(status_code=404, detail="No image_labeling rows found")
        return {"status": "success", "data": row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ---------------- Dyslexia Myths ---------------- #
@app.post("/myth/next")
def myth_next():
    try:
        rows = fetch_next_myth_row(batch_size=10)
        if not rows:
            raise HTTPException(status_code=404, detail="No myth rows found")
        data = [dict(r) for r in rows]
        return {"status": "success", "count": len(data), "data": data}
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing DB config: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ---------------- Parent Chatbot ---------------- #
class ParentChatRequest(BaseModel):
    question: str
    kb_hit: Optional[str] = None


@app.post("/parent_chat")
def parent_chat(req: ParentChatRequest):
    api_key = get_gcv_api_key()
    try:
        result = get_parent_answer(req.question, req.kb_hit, api_key)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")
