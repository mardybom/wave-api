# Wave – Alphabet Mastery API

### A modular FastAPI backend powering the *Wave* learning platform for dyslexic learners.

This backend provides interactive, AI-powered features to support literacy and spelling through short, engaging activities.  
It integrates handwriting recognition, sentence rearranging, image labeling, myth clarification, and a parent-friendly chatbot.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Framework | FastAPI (Python 3.10+) |
| Database | PostgreSQL (Neon Cloud) |
| DB Access | psycopg2 |
| Vision OCR | Google Cloud Vision API |
| AI Chatbot | Google Gemini (Grounding with Search) |
| Speech / Phonics | pronouncing (ARPAbet) |
| Deployment | Azure App Service |

---

## Project File Structure

```
wave-api/
├── main.py                     # FastAPI entry point
├── gcv_config.py               # Google Vision key management
├── db_config.py                # Database connection
├── canvas_detector.py          # Handwriting recognition logic
├── sentence_rearranging.py     # Sentence sequencing logic
├── image_labeling.py           # Image labeling + ARPAbet helpers
├── dyslexia_myths.py           # Myth/truth rotation
├── chatbot.py                  # Gemini-powered parent chatbot
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Environment Variables

We had set these in our `.env` file or Azure App Service → Configuration.

| Variable | Description |
|-----------|-------------|
| `GCV_API_KEY` | Google Cloud Vision / Gemini API key |
| `DB_HOST` | Neon database host |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `DB_PORT` | 5432 |
| `DB_SSLMODE` | `require` (for Neon) |

---

## Setup & Run Locally

```bash
# 1. Clone
git clone https://github.com/yourusername/wave-api.git
cd wave-api

# 2. Create virtual environment for python
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables locally
export GCV_API_KEY="your_key"
export DB_HOST="your_neon_host"
export DB_NAME="wave_db"
export DB_USER="wave_app"
export DB_PASSWORD="your_password"
export DB_PORT=5432
export DB_SSLMODE=require

# 5. Run FastAPI
fastapi dev main.py
```

Visit **http://127.0.0.1:8000/docs** for the interactive UI.

---

## Deployment (Azure)

1. Push code to GitHub.  
2. Create an Azure App Service → Python 3.10.  
3. Set all environment variables under *Configuration → Application Settings*.  
4. Deploy via GitHub Actions or zip upload.  
5. Verify via `https://<your-app>.azurewebsites.net/docs`.

---

## API Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/alphabet_mastery` | POST | Verify handwritten letter (OCR) |
| `/sentence/next` | POST | Retrieve next sentence by difficulty |
| `/image_labeling/next` | POST | Fetch random image + fake labels |
| `/myth/next` | POST | Get rotating myths/truths |
| `/parent_chat` | POST | Ask the parent-friendly chatbot |

---

## Feature Details

### **Alphabet Mastery**
- Accepts base64 canvas image + expected letter.  
- Uses Google Vision OCR for character recognition.  
- Returns confidence and match ratio to guide learner feedback.

### **Sentence Rearranging**
- Provides unique sentence per difficulty level.  
- Uses PostgreSQL cursor tracking to avoid repetition.

### **Image Labeling**
- Returns random labeled image with rearranged word options.  
- Generates ARPAbet phonetic transcription for speech support.

### **Dyslexia Myths**
- Cycles through myth/truth pairs for awareness activities.  
- Cursor system ensures smooth wrap-around pagination.

### **Parent Chatbot**
- Grounded answers for parents of learners with dyslexia.  
- Empathetic, non-diagnostic, with citations from trusted sources.

---

## Testing

Run unit tests (pytest recommended):

```bash
pytest --maxfail=1 --disable-warnings -q
```

Coverage goals: ≥ 95 % across major modules (`canvas_detector.py`, `chatbot.py`, etc.).

---

## Security Notes

- CORS restricted to trusted frontend domains.  
- Secrets handled via Azure environment variables.  
- Database and API connections use SSL.  
- No hard-coded keys or credentials.

---

## Author

**Aadhithyanarayanan V A**
**Ling Tang**

Monash University — FIT5120 Industry Experience Project  

Industry Mentor         : Divya and Anant
Mentor         : Himanshu Jethanandani
Tech Mentor         : Ryan Pathirana

---

> *“Empowering learners through inclusive, intelligent, and evidence-based technology.”*
