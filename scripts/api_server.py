import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

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

# ─── Run with `python scripts/api_server.py` ───────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
