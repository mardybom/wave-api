from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from canvas_detector import detect_handwritten_letters_from_base64, CanvasInput
import os  
app = FastAPI()

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

@app.post("/alphabet_mastery")
def read_canvas_input(request: CanvasInput):
    api_key = os.getenv("GCV_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GCV_API_KEY configuration")

    try:
        detected_letters = detect_handwritten_letters_from_base64(request.canvas_input, api_key)
        return {
            "status": "success",
            "detected_count": len(detected_letters),
            "letters": detected_letters
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")