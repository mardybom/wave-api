from typing import Union

from fastapi import FastAPI, HTTPException

from canvas_detector import detect_handwritten_letters_from_base64, CanvasInput

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/alphabet_mastery")
def read_canvas_input(request: CanvasInput):
    """
    Detect handwritten letters from base64 canvas input
    
    Args:
        request: CanvasInput object containing base64 image
        
    Returns:
        JSON object with detected letters and confidence levels
    """
    
    # Your Google Cloud Vision API key
    api_key = "AIzaSyCZ0xAjR_jrk-vdWaEP-KwjNRBqK9XVZpc"
    
    try:
        detected_letters = detect_handwritten_letters_from_base64(
            request.canvas_input, 
            api_key
        )
        
        return {
            "status": "success",
            "detected_count": len(detected_letters),
            "letters": detected_letters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")





