"""
Model utilities for loading tokenizers and models.
Also will include custom Trainer and metrics for fine-tuning transformers.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default model name mapping for convenience
_model_name_map = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "longformer": "allenai/longformer-base-4096"
}

def get_tokenizer(model_name: str):
    """
    Get a HuggingFace tokenizer for the specified model name.
    Args:
        model_name (str): Name or key of the model (e.g., 'bert', 'roberta', or full HF model name).
    Returns:
        PreTrainedTokenizer: The tokenizer instance.
    """
    # Map shorthand names to full model identifiers if needed
    hf_name = _model_name_map.get(model_name.lower(), model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return tokenizer

def get_model(model_name: str, num_labels: int):
    """
    Get a HuggingFace AutoModelForSequenceClassification for the specified model.
    Args:
        model_name (str): Name or key of the model (as in get_tokenizer).
        num_labels (int): Number of output labels for classification.
    Returns:
        PreTrainedModel: The loaded model ready for classification.
    """
    hf_name = _model_name_map.get(model_name.lower(), model_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=num_labels)
    return model
