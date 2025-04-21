"""
Model utilities for loading tokenizers and models.
Also will include custom Trainer and metrics for fine-tuning transformers.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
import torch
import torch.nn.functional as F


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


class CustomTrainer(Trainer):
    """
    Custom Trainer that allows using weighted loss or focal loss during training.
    """
    def __init__(self, use_focal=False, alpha=None, gamma=2.0, *args, **kwargs):
        """
        Args:
            use_focal (bool): If True, use focal loss; if False, use standard cross-entropy.
            alpha (list or torch.Tensor): Class weight coefficients for imbalance (len = num_labels).
            gamma (float): Focusing parameter for focal loss.
        """
        super().__init__(*args, **kwargs)
        self.use_focal = use_focal
        # Convert alpha to tensor if provided (for weighted loss)
        if alpha is not None:
            self.class_weights = torch.tensor(alpha, dtype=torch.float)
        else:
            self.class_weights = None
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to apply weighted cross-entropy or focal loss.
        """
        labels = inputs.get("labels")
        # Forward pass (do not pass labels to model to compute raw logits)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        # Compute standard cross-entropy loss (with optional class weights)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
        if self.use_focal:
            # Compute softmax probabilities for focal loss
            probs = F.softmax(logits, dim=1)
            # Get the probabilities of the true class for each sample
            pt = probs[range(len(labels)), labels]
            # Focal loss scaling factor ( (1 - p_t)^gamma )
            focal_factor = (1 - pt) ** self.gamma
            loss = (focal_factor * ce_loss).mean()
        else:
            # Standard weighted cross-entropy
            loss = ce_loss.mean()
        return (loss, outputs) if return_outputs else loss
