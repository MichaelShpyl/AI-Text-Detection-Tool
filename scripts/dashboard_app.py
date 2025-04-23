# File: scripts/dashboard_app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AI Text Detector Dashboard"

# Helper to load and encode images
def encode_image(image_file):
    path = os.path.join(os.path.dirname(__file__), '..', 'diagrams', image_file)
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

# Preload images
class_dist_img    = encode_image("class_distribution.png")
length_dist_img   = encode_image("length_distribution.png")
conf_matrix_img   = encode_image("confusion_matrix.png")
roc_curves_img    = encode_image("roc_curves.png")

# Layout
app.layout = html.Div([
    html.H1("AI-Generated Text Detection Dashboard", style={"textAlign": "center", "marginBottom": "1em"}),
    dcc.Tabs(id="tabs", value="tab-eda", children=[
        dcc.Tab(label="EDA",        value="tab-eda"),
        dcc.Tab(label="Evaluation", value="tab-eval"),
        dcc.Tab(label="Inference",  value="tab-inf"),
    ]),
    html.Div(id="tab-content")
])

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-eda":
        return html.Div([
            html.H3("Exploratory Data Analysis", style={"textAlign": "center", "marginTop": "1em"}),
            html.Img(src=class_dist_img,  style={"width":"45%","display":"inline-block","padding":"1em"}),
            html.Img(src=length_dist_img, style={"width":"45%","display":"inline-block","padding":"1em"}),
            html.P("Dataset is balanced. Human-written → longer than AI-generated/paraphrased.",
                   style={"textAlign":"center","fontStyle":"italic","marginTop":"0.5em"})
        ])
    elif tab == "tab-eval":
        return html.Div([
            html.H3("Model Evaluation", style={"textAlign": "center", "marginTop": "1em"}),
            html.Img(src=conf_matrix_img, style={"width":"40%","display":"inline-block","padding":"1em"}),
            html.Img(src=roc_curves_img,  style={"width":"50%","display":"inline-block","padding":"1em"}),
            html.P("Accuracy ~91%. Great at human vs AI; confuses AI-para vs AI-gen.",
                   style={"textAlign":"center","fontStyle":"italic","marginTop":"0.5em"})
        ])
    else:  # tab-inf
        return html.Div([
            html.H3("Try the Detector", style={"textAlign":"center","marginTop":"1em"}),
            dcc.Textarea(id="input-text", placeholder="Enter text...", style={"width":"80%","height":"120px"}),
            html.Br(),
            html.Button("Detect", id="detect-button", n_clicks=0, style={"marginTop":"0.5em"}),
            html.Div(id="result-output", style={"marginTop":"1em"})
        ])

# Inference callback (same as in the notebook)
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_path = os.path.join(os.path.dirname(__file__), '..', 'diagrams', 'final_model')
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
labels     = ["Human-written", "AI-paraphrased", "AI-generated"]

@app.callback(Output("result-output", "children"),
              Input("detect-button", "n_clicks"),
              Input("input-text",     "value"))
def run_detection(nc, txt):
    if not nc or not txt:
        return ""
    toks = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**toks).logits
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs)); lbl = labels[idx]; conf = probs[idx]
    # Simple output
    return html.Div([
        html.H4(f"Prediction: {lbl}", style={"textAlign":"center"}),
        html.P(f"Confidence: {conf*100:.1f}%", style={"textAlign":"center"})
    ])

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
