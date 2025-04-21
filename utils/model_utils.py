import torch
import torch.nn as nn
from transformers import Trainer

def focal_loss(logits, labels, alpha=None, gamma=2):
    """
    Compute focal loss for multi-class classification.
    logits: model outputs (before softmax).
    labels: true labels.
    alpha: class weight tensor (for addressing class imbalance).
    gamma: focusing parameter (int).
    """
    ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none', weight=alpha)
    # Get the probability of the true class
    pt = torch.softmax(logits, dim=-1)[range(len(labels)), labels]
    # Focal loss scaling factor
    focal_factor = (1 - pt) ** gamma
    loss = focal_factor * ce_loss
    return loss.mean()

class CustomTrainer(Trainer):
    """
    Custom Trainer to incorporate class-weighted and focal loss.
    """
    def __init__(self, use_focal=False, alpha=None, gamma=2, **kwargs):
        """
        use_focal: if True, use focal loss; if False, use standard CrossEntropy.
        alpha: list or tensor of class weights (for cross-entropy or focal loss alpha).
        gamma: focusing parameter for focal loss.
        """
        super().__init__(**kwargs)
        self.use_focal = use_focal
        self.gamma = gamma
        if alpha is not None:
            self.class_weights = torch.tensor(alpha, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to apply weighted cross-entropy or focal loss.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.use_focal:
            alpha = None
            if self.class_weights is not None:
                alpha = self.class_weights.to(logits.device)
            loss = focal_loss(logits, labels, alpha=alpha, gamma=self.gamma)
        else:
            if self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
