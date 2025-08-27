from pydantic import BaseModel
from fastapi import HTTPException
from google.cloud import vision
import base64

# Request model
class CanvasInput(BaseModel):
    canvas_input: str  # base64 encoded image

def detect_handwritten_letters_from_base64(base64_image: str, api_key: str):
    """
    Detect individual handwritten letters from base64 image using Google Cloud Vision API
    
    Args:
        base64_image (str): Base64 encoded image string
        api_key (str): API key for authentication
        
    Returns:
        list: List of detected letters with confidence levels
    """
    
    try:
        # Initialize the client with API key
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        
        # Decode base64 image
        # Remove data URL prefix if present (data:image/png;base64,)
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
            
        image_data = base64.b64decode(base64_image)
        
        # Create Vision API image object
        image = vision.Image(content=image_data)
        
        # Perform document text detection with language hints
        image_context = vision.ImageContext(language_hints=['en'])  # Force English/Latin
        response = client.document_text_detection(image=image, image_context=image_context)
        
        if response.error.message:
            raise Exception(f'{response.error.message}')
            
        document = response.full_text_annotation
        detected_letters = []
        
        # Process the single image (no page loop needed for canvas image)
        if document.pages:
            for block in document.pages[0].blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            letter = symbol.text
                            confidence = symbol.confidence
                            
                            # Only keep Latin alphabetic characters
                            if letter.isalpha() and ord(letter) < 128:  # ASCII letters only
                                detected_letters.append({
                                    'letter': letter,
                                    'confidence': round(confidence, 3)
                                })
        
        return detected_letters
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")