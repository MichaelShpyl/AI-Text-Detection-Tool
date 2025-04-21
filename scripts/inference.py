"""
Command-line inference script.
Allows users to input text and get a prediction from the trained model.
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description="AI Text Detector Inference")
    parser.add_argument('--model-dir', type=str, default="diagrams/final_model",
                        help="Path to the directory containing the fine-tuned model (tokenizer and model files).")
    parser.add_argument('--text', type=str, required=True,
                        help="Input text to analyze (enclose in quotes).")
    parser.add_argument('--output-json', type=str, default=None,
                        help="File path to save the prediction result in JSON format.")
    args = parser.parse_args()

    # Placeholder: Loading model and performing inference will be added in next commit.

    print(f"Model directory: {args.model_dir}")
    print(f"Input text: {args.text[:50]}{'...' if len(args.text) > 50 else ''}")
    if args.output_json:
        print(f"Results will be saved to {args.output_json}")

if __name__ == "__main__":
    main()
