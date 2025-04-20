# EXAMINER GUIDE

This guide provides instructions to set up, run, and evaluate the **AI Text Detector** project, which distinguishes between human-written, AI-paraphrased, and AI-generated text. Follow the steps below to reproduce our results and interact with the trained models.

## Project Structure

- **config.yaml** – Configuration file with paths, training hyperparameters, and feature toggles.
- **requirements.txt** – Python dependencies with specific versions for reproducibility.
- **EXAMINER_GUIDE.md** – (This guide) Instructions for setup and execution.
- **logs/** – Directory containing logs (e.g., `training.log` for model training output).
- **data/** – Directory for datasets. By default, expects `final_dataset.csv` here. Also contains `modern_articles/` for future data.
- **notebooks/** – Jupyter notebooks documenting each step of the project:
  - `01_data_prep.ipynb` – Data loading, cleaning, and preparation.
  - `02_eda.ipynb` – Exploratory Data Analysis.
  - `03_feature_engineering.ipynb` – Feature extraction beyond raw text.
  - `04_model_training.ipynb` – Model training and hyperparameter tuning.
  - `05_evaluation.ipynb` – Model evaluation and interpretation.
  - `06_dashboard.ipynb` – Building interactive dashboards for the model.
- **utils/** – Utility modules containing reusable code (data processing, model utils, etc.).
- **scripts/** – Standalone scripts (e.g., `inference.py`, and dashboard apps) for command-line or web app interaction.
- **diagrams/** – Directory where figures, plots, and model artifacts (e.g., saved model) are stored.

## Setup Instructions

1. **Environment Setup**: Ensure you have **Python 3.x** installed. Create a virtual environment (optional but recommended) and activate it.
2. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required packages. This may take a few minutes, as it includes large libraries like Transformers and PyTorch.
   - All package versions are pinned for consistency. If you encounter version conflicts, use the provided versions.
   - *Note:* If you plan to use the optional features (like spaCy for NER), install those separately (e.g., `pip install spacy` and download the model with `python -m spacy download en_core_web_sm`).
3. **Weights & Biases (Optional)**: If you want to enable experiment tracking with Weights & Biases, log in by setting your API key. For example, run `wandb login` in the terminal or set the environment variable `WANDB_API_KEY` with your key. If you skip this, the code will default to logging training metrics locally (e.g., to TensorBoard).

## Data Placement

- **Main Dataset**: Place your dataset file in the `data/` folder. By default, the code expects a file named **`final_dataset.csv`** containing the text samples and labels. If your data is in multiple files or a different format, you may need to adjust the loading code or combine them into one CSV. The structure assumed by the code is either:
  - A single column (e.g., "text") and a label column (e.g., "label"), **OR**
  - Three columns, one for each class of text (e.g., "human_written", "ai_paraphrased", "ai_generated"), where each row contains parallel texts. The code will flatten this into a single text+label list.
- **Additional Data**: If you have **modern articles or future datasets** to test the model on, place them in the `data/modern_articles/` directory. This is not used in training by default, but you can load and analyze these later (we provide stub code for this in the notebooks).

Make sure to update **`config.yaml`** if your file paths or names differ from the defaults.

## Running the Notebooks

Open the project Jupyter notebooks in the given order to reproduce the analysis:

- **1️⃣ 01_data_prep.ipynb**: This notebook detects the environment (GPU/TPU/CPU), loads the dataset, cleans the text, and splits the data into train/validation/test sets. **Run all cells in order.** By the end, you will have a prepared dataset ready for analysis and modeling.
- **2️⃣ 02_eda.ipynb**: Explore the data through visualizations and summary statistics. We examine class distributions, text length, readability scores, etc. Run this to understand differences between human and AI-generated text prior to modeling.
- **3️⃣ 03_feature_engineering.ipynb**: Calculate additional features that might help differentiate the texts, such as readability indices, sentiment scores, lexical diversity metrics, and more. This notebook justifies each feature and records observations about their potential usefulness.
- **4️⃣ 04_model_training.ipynb**: Fine-tune multiple transformer models (BERT, RoBERTa, Longformer) on the training data. We experiment with different loss functions (weighted cross-entropy vs focal loss) and perform hyperparameter tuning using Optuna (with Weights & Biases logging if enabled). **⚠️ Runtime:** Training large models on a full dataset (hundreds of thousands of samples) can be time-consuming. We recommend using a GPU; on a modern GPU, each epoch for BERT-base may take ~5-10 minutes (on CPU it could be hours). In the notebook, we provide options to adjust training for quicker execution if needed.
- **5️⃣ 05_evaluation.ipynb**: Evaluate the best model on the test set. This includes computing precision, recall, F1-score for each class, plotting the confusion matrix and ROC/PR curves, and using SHAP or LIME to interpret model predictions. We also discuss error analysis and which types of texts are hardest to classify.
- **6️⃣ 06_dashboard.ipynb**: Demonstrate how to deploy the trained model in interactive dashboards (Plotly Dash and Streamlit). We provide example code for a Dash app and a Streamlit app that allow you to input custom text and see the model’s prediction. We also implement functionality to save user session results (e.g., download predictions as a CSV).

Run each notebook in sequence. The notebooks contain rich markdown explanations in a first-person academic tone, describing each step and rationale. They also include "**Mistakes & Fixes**" callouts highlighting how we addressed various pitfalls during development (e.g., data leakage, class imbalance handling) and "**Runtime warnings**" for time-consuming steps.

## Launching the Dash and Streamlit Apps

After training and evaluating the model, you can interact with it through the provided web apps:

- **Dash App (Plotly Dash)**: We provide the app code in `scripts/dashboard_app.py`. To run it, execute `python scripts/dashboard_app.py` in your terminal. This will launch a local web server (by default at http://127.0.0.1:8050/). Open that address in a browser to access the dashboard. The Dash app includes:
  - Tabs for **EDA visuals** (loading plots from the `diagrams/` directory, such as distribution charts and ROC curves) so you can see the findings graphically.
  - A **Text Input** section where you can enter any text and the app will classify it in real-time as Human, AI-Paraphrased, or AI-Generated.
  - A feature to save results: for example, it keeps a log of texts you test and their predictions, which you can copy or download.
- **Streamlit App**: Alternatively, run `streamlit run scripts/streamlit_app.py`. This opens a Streamlit interface in your web browser. The Streamlit app is a simpler UI:
  - You can enter or paste text into a text box, and upon submission, the predicted class is displayed.
  - For each input, a record is stored (in-memory) and a **"Download Results"** button allows you to download a CSV of all inputs and predictions from your session.
  - This app is handy for rapidly testing multiple examples in a user-friendly way.

Both dashboards utilize the same trained model (by loading the saved model from `diagrams/final_model/`). They demonstrate how the model can be deployed for end-users. Feel free to experiment with both to see which suits your needs.

## Using the CLI Inference Script

For quick predictions without using Jupyter or a web app, use the provided command-line script:

```bash
python scripts/inference.py --text "Your text to analyze goes here."
