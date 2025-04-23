## Running the AI Text Detector Project

1. **Backend API**: Navigate to the `scripts/` directory and run `python api_server.py`. This will load the trained model and start the FastAPI server at `http://127.0.0.1:8000`. Keep this terminal open.

2. **Dashboard (Plotly Dash)**: To view the Plotly Dash dashboard, open `notebooks/06_dashboard_dash.ipynb` in Jupyter and run all cells, or run the script `python scripts/dashboard_app.py` if provided. This will launch a Dash app on `localhost:8050` where you can navigate through EDA, Evaluation, and Inference tabs. (Ensure the backend is running if you want to use the inference tab via the API, or use the built-in model in Dash.)

3. **React Frontend**: Navigate to `frontend/` and run `npm install` (first time) and `npm start`. This opens the React application at `http://localhost:3000`. Use the sidebar to switch between single text analysis and batch file upload. Try entering custom text or uploading sample files (in `data/` or your own) to see results. The frontend will call the local API to get predictions, so the backend must be running.

4. **Chrome Extension**: Load the extension by going to Chrome's Extensions page, enabling Developer Mode, and clicking "Load Unpacked". Select the `extension/` folder. Then, visit any article or web page. A blue "Scan this article" button will appear at the bottom-right. Click it to analyze the page's text. A small panel will show the predicted class and confidence, and the page text will be highlighted with yellow for words that influenced the decision. (Make sure the backend API is running before scanning.)

## Design Justification and Usage

- The **Plotly Dash dashboard** (Notebook 06) provides an interactive report for examiners. It includes visualizations of the data distribution and model performance. The Inference tab in Dash allows quick testing of the model on custom text and uses LIME to highlight important words directly in the app.

- The **React frontend** offers a richer user experience, with a responsive design (sidebar navigation, mobile-friendly drawer, and dark mode) making it suitable for end-user interaction. The Single Text Analysis page not only shows the prediction but also a breakdown of probabilities and highlights which words in the input were most influential. The Batch Upload page supports drag-and-drop of multiple files (txt, docx, pdf, html), which are processed sequentially and results displayed in a table. This is ideal for processing a batch of documents in one go.

- The **Chrome extension** extends functionality to the web at large. It demonstrates how the detector can be used in real time on any website. The decision to highlight text in situ helps visualize what cues the model picked up on a real article. This is a powerful demonstration of the model's explainability and practicality: an examiner can go to a news site, click scan, and see immediate insight into that article's origin (human or AI).

All components run **locally**. No internet connection is needed beyond downloading dependencies. This ensures privacy (documents or text stay on your machine) and aligns with an exam setting where external services might not be permitted.

### Example Use Case:
Imagine you have a set of articles and you suspect some were AI-generated. You can use the Batch Upload tool to scan all the articles at once – the table will quickly show you which ones are likely AI. For a specific article, you can then use the Single Analysis or the Chrome extension to delve deeper, highlighting exactly which phrases led the model to tag it as AI-generated. The Dash dashboard’s EDA and Evaluation sections will help you explain to the examiner how the model was trained and how well it performs (e.g., ~91% accuracy, with most errors between AI-paraphrased vs AI-generated texts).

## Conclusion

I have built a comprehensive AI Text Detector system comprising:
- A robust backend with a fine-tuned RoBERTa model and explainability via LIME.
- Multiple frontends (Dash for analysis/report, React for user-friendly operation, Chrome extension for on-page use) to interact with the model.
- Support for various input types and scenarios (direct text, batch files, live webpages).
This modular design and the included documentation aim to make it easy for examiners to run and evaluate the project. Follow the steps above to explore each component.
