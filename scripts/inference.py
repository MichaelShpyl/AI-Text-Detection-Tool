"""
Command-line inference script.
Allows users to input text and get a prediction from the trained model.
"""
import argparse
import json
from utils import dashboard_utils

def main():
    parser = argparse.ArgumentParser(description="AI Text Detector Inference")
    parser.add_argument('--model-dir', type=str, default="diagrams/final_model",
                        help="Path to the directory containing the fine-tuned model (tokenizer and model files).")
    parser.add_argument('--text', type=str, required=True,
                        help="Input text to analyze.")
    parser.add_argument('--output-json', type=str, default=None,
                        help="File path to save the prediction result as JSON.")
    args = parser.parse_args()

    # Load the tokenizer and model from the specified directory
    tokenizer = dashboard_utils.AutoTokenizer.from_pretrained(args.model_dir)
    model = dashboard_utils.AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # Run inference
    label, probs = dashboard_utils.predict_text(args.text, tokenizer, model)

    # Prepare result dict
    result = {
        "input_text": args.text,
        "predicted_label": label,
        "class_probabilities": probs
    }
    # Print to console
    print(json.dumps(result, indent=2))

    # Optionally save to JSON file
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {args.output_json}")

if __name__ == "__main__":
    main()
