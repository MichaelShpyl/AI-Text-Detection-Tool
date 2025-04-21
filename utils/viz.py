"""
Visualization helpers for data exploration and evaluation results.
Provides plotting functions using matplotlib and seaborn.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_class_distribution(df, label_col='label', save_path=None):
    """
    Plot a bar chart of class distribution.
    Optionally save the figure to a file.
    Returns:
        matplotlib.figure.Figure: The figure object for further manipulation if needed.
    """
    counts = df[label_col].value_counts()
    classes = counts.index.tolist()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=classes, y=counts.values, ax=ax, palette='pastel')
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha='center')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

def plot_length_distribution(df, text_col='text', label_col='label', save_path=None):
    """
    Plot the distribution of text lengths for each class.
    Creates overlapping histograms colored by class.
    Optionally save the figure to a file.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Compute text lengths
    df['__length__'] = df[text_col].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(figsize=(6,4))
    classes = df[label_col].unique()
    for cls in classes:
        lengths = df[df[label_col] == cls]['__length__']
        sns.histplot(lengths, ax=ax, bins=30, label=str(cls), element="step", fill=False)
    ax.set_title("Article Length Distribution by Class")
    ax.set_xlabel("Length (words)")
    ax.set_ylabel("Count")
    ax.legend(title="Class")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    # Clean up the temporary column
    df.drop(columns=['__length__'], inplace=True, errors='ignore')
    return fig


def plot_confusion_matrix(y_true, y_pred, labels, normalize=False, save_path=None):
    """
    Plot a confusion matrix given true and predicted labels.
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of label names (for axes).
        normalize (bool): If True, normalize counts to percentages.
        save_path (str): File path to save the figure (optional).
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """
    Plot ROC curves for each class (one vs rest).
    Args:
        y_true (array-like): True labels (integers).
        y_prob (ndarray): Predicted probabilities (shape: n_samples x n_classes).
        class_names (list): Names of classes for labeling.
        save_path (str): File path to save the figure (optional).
    Returns:
        matplotlib.figure.Figure: The ROC curves figure.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    n_classes = len(class_names)
    for i in range(n_classes):
        # Binarize labels for class i vs rest
        true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

def plot_pr_curves(y_true, y_prob, class_names, save_path=None):
    """
    Plot Precision-Recall curves for each class (one vs rest).
    """
    fig, ax = plt.subplots(figsize=(6,5))
    n_classes = len(class_names)
    for i in range(n_classes):
        true_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(true_binary, y_prob[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{class_names[i]} (AUC = {pr_auc:.2f})")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


