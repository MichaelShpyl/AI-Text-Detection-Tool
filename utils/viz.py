"""
Visualization helpers for data exploration and evaluation results.
Provides plotting functions using matplotlib and seaborn.
"""
import matplotlib.pyplot as plt
import seaborn as sns

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
