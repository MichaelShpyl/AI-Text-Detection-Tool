"""
Utilities shared by dashboard applications (Dash and Streamlit).
Includes model loading and inference for new inputs.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
import numpy as np
from lime.lime_text import LimeTextExplainer
import json
import datetime

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


def explain_prediction(text, tokenizer, model, num_features=6):
    """
    Generate an explanation for the model's prediction on the given text using LIME.
    Returns:
        list of (str, float): Top contributing words and their weights for the predicted class.
    """
    # Define a prediction function for LIME that returns class probabilities
    def _predict_proba(texts):
        results = []
        for t in texts:
            _, probs = predict_text(t, tokenizer, model)
            # Return probabilities in the order of class_names
            results.append([probs[_label_map[i]] for i in range(len(_label_map))])
        return np.array(results)

    # Use LIME to explain the prediction for this single text
    exp = _explainer.explain_instance(text, _predict_proba, num_features=num_features, labels=[0, 1, 2])
    # Get the predicted class index
    pred_label, probs = predict_text(text, tokenizer, model)
    pred_idx = int(torch.argmax(torch.tensor(list(probs.values()))))
    # Get explanation weights for the predicted class
    explanation = exp.as_list(label=pred_idx)
    return explanation


def save_session_entry(text, predicted_label, mode='json'):
    """
    Save a record of the input text and model prediction to a session log.
    Args:
        text (str): The input text analyzed.
        predicted_label (str): The model's predicted class for the text.
        mode (str): 'json' or 'csv' indicating the format (Dash uses JSON, Streamlit uses CSV).
    """
    timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    if mode == 'json':
        log_path = config['paths']['session_log_json']
        entry = {"timestamp": timestamp, "text": text, "predicted_label": predicted_label}
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(entry)
        with open(log_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif mode == 'csv':
        log_path = config['paths']['session_log_csv']
        header = "timestamp,text,predicted_label\n"
        # Make sure newlines in text don’t break CSV rows
        safe_text = text.replace('\n', ' ').replace('\r', ' ')
        line = f"{timestamp},\"{safe_text}\",{predicted_label}\n"
        try:
            with open(log_path, 'x') as f:  # create file if not exists
                f.write(header)
                f.write(line)
        except FileExistsError:
            with open(log_path, 'a') as f:
                f.write(line)