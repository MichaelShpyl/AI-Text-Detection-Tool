"""
Utilities shared by dashboard applications (Dash and Streamlit).
Includes model loading and inference for new inputs.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml

# Load config to get model path
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
_final_model_dir = config['paths']['model_dirs']['final']

# Load class label mapping for decoding predictions
_label_map = {v: k for k, v in config['model']['label_mapping'].items()}

def load_final_model():
    """
    Load the fine-tuned final model and its tokenizer from disk.
    Returns:
        tokenizer, model: The loaded tokenizer and model ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(_final_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(_final_model_dir)
    model.eval()  # set model to evaluation mode
    return tokenizer, model

def predict_text(text, tokenizer, model):
    """
    Predict the class of a given text using the provided tokenizer and model.
    Returns:
        tuple: (predicted_label_name, confidences), where confidences is a dict of class probabilities.
    """
    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config['training']['max_length']['bert_roberta']
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].tolist()
    # Determine predicted class index
    pred_idx = int(torch.argmax(logits, dim=1))
    # Map to class name
    label_name = _label_map.get(pred_idx, str(pred_idx))
    # Create a dictionary of class probabilities
    class_probs = { _label_map[i]: float(probs[i]) for i in range(len(probs)) }
    return label_name, class_probs
