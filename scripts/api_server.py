import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
from fastapi import UploadFile, File
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO

# ─── Initialize FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="AI Text Detector API")

# ─── CORS (allow all origins for local development) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # in prod, restrict to my front-end origin(s)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model & tokenizer from local `diagrams/final_model` ─────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "diagrams" / "final_model"
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Could not find model folder at {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)
model.eval()

label_names = ["Human-written", "AI-paraphrased", "AI-generated"]

# ─── Pydantic schema for incoming JSON ────────────────────────────────────
class TextRequest(BaseModel):
    text: str

# ─── Single-text prediction endpoint ──────────────────────────────────────
@app.post("/predict")
async def predict_text(req: TextRequest):
    text = req.text
    # Tokenize and run the model
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    pred_idx   = int(torch.argmax(logits, dim=1).item())
    pred_label = label_names[pred_idx]
    confidence = probs[pred_idx]

    # Attempt LIME explanation (if available)
    try:
        from utils import dashboard_utils
        explanation_pairs = dashboard_utils.explain_prediction(
            text, tokenizer, model, num_features=6
        )
        explanation = [
            {"word": w, "weight": float(weight)}
            for (w, weight) in explanation_pairs
        ]
    except Exception:
        explanation = []

    # Build probabilities dict
    probabilities = {
        label_names[i]: probs[i] for i in range(len(label_names))
    }

    return {
        "prediction":   pred_label,
        "confidence":   confidence,
        "probabilities": probabilities,
        "explanation":  explanation
    }

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded file (txt, html, docx, or pdf). Extracts text and returns prediction results.
    """
    # Read file contents into memory
    contents = await file.read()
    filename = file.filename.lower()
    text_content = ""
    try:
        if filename.endswith(".txt"):
            # Decode bytes to text
            text_content = contents.decode('utf-8', errors='ignore')
        elif filename.endswith(".docx"):
            # Use python-docx to read text
            from io import BytesIO
            from docx import Document
            doc = Document(BytesIO(contents))
            text_content = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(".html") or filename.endswith(".htm"):
            # Parse HTML and extract visible text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(contents, "html.parser")
            text_content = soup.get_text(separator=" ")
        elif filename.endswith(".pdf"):
            # Try extracting text from PDF using PyMuPDF
            import fitz  # PyMuPDF
            pdf = fitz.open(stream=contents, filetype="pdf")
            for page in pdf:
                text_content += page.get_text()
            pdf.close()
            # If no text extracted (scanned PDF), use OCR
            if text_content.strip() == "":
                from pdf2image import convert_from_bytes
                import pytesseract
                images = convert_from_bytes(contents)
                for img in images:
                    text_content += pytesseract.image_to_string(img)
        else:
            return {"error": "Unsupported file type"}
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

    text_content = text_content.strip()
    if text_content == "":
        return {"error": "No text found in the document"}

    # Reuse prediction logic from /predict
    inputs = tokenizer(text_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy().tolist()
    pred_idx = int(torch.argmax(outputs.logits, dim=1).item())
    pred_label = label_names[pred_idx]
    confidence = probs[pred_idx]
    return {
        "prediction": pred_label,
        "confidence": confidence
        # (I omit the explanation here for efficiency – running LIME on a long document could be time-consuming. Batch analysis typically focuses on classification results; the user can always analyze a specific excerpt via the single-text route to get highlights.)
    }

# ─── Run with `python scripts/api_server.py` ───────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
