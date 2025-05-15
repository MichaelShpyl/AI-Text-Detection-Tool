[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![Node.js 14+](https://img.shields.io/badge/Node.js-14%2B-green.svg)](https://nodejs.org/)

# AI Text Detection Tool

A modular, full-stack system for detecting AI-generated text in news and academic essays.  
Supports API, dashboards, web apps, and a Chrome extension for real-time scanning and explainability.

---

## Table of Contents
- [AI Text Detection Tool](#ai-text-detection-tool)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Project](#running-the-project)
    - [Backend API](#backend-api)
    - [Dashboard (Plotly Dash)](#dashboard-plotly-dash)
    - [React Frontend](#react-frontend)
    - [Chrome Extension](#chrome-extension)
  - [API Usage](#api-usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [Roadmap](#roadmap)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgments](#acknowledgments)

---

## Features
- **FastAPI Backend** with a fine-tuned RoBERTa model for high-accuracy inference.
- **LIME Explainability** highlights which words influenced each prediction.
- **Plotly Dash Dashboard** for interactive EDA, model evaluation, and inference.
- **React Web App** for single-text and batch-file analysis with a responsive UI.
- **Chrome Extension** enabling on-page scanning of any web content.

---

## Prerequisites
- **Python** 3.10 or higher  
- **Node.js** 14.x or higher + npm  
- Optional: **Docker** & **Docker Compose** (for containerized setup)

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/MichaelShpyl/AI-Text-Detection-Tool.git
   cd AI-Text-Detection-Tool
   ```

2. **Create & activate a Python virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate.bat   # Windows
   ```

3. **Install Python dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**  
   ```bash
   cd frontend
   npm install
   cd ..
   ```

---

## Configuration

- **`.env` file** (optional): You can create a `.env` in the project root with:
  ```ini
  BACKEND_HOST=127.0.0.1
  BACKEND_PORT=8000
  MODEL_PATH=./models/roberta_ai_detector.pt
  ```

- **Dash settings**: By default, the Dash app will connect to `http://127.0.0.1:8000`. Adjust in `scripts/dashboard_app.py` if needed.

---

## Running the Project

### Backend API
```bash
cd scripts
python api_server.py
```
- Starts FastAPI at `http://127.0.0.1:8000`
- Loads the trained model into memory

---

### Dashboard (Plotly Dash)
- **Option A – Notebook**  
  Open `notebooks/06_dashboard_dash.ipynb` in Jupyter and **Run All**.
- **Option B – Script**  
  ```bash
  python scripts/dashboard_app.py
  ```
  - Dashboard runs at `http://localhost:8050`  
  - Tabs: EDA, Evaluation, Inference (API-backed)

---

### React Frontend
```bash
cd frontend
npm start
```
- Opens at `http://localhost:3000`  
- Sidebar: **Single Text Analysis** | **Batch Upload**

---

### Chrome Extension
1. Go to `chrome://extensions/`, enable **Developer mode**  
2. Click **Load unpacked**, select the `extension/` folder  
3. Visit any webpage, click **Scan this article**  
4. View predictions & highlights (ensure API is running)

---

## API Usage

```bash
# Single-text inference
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"text": "Sample input text to analyze."}'
```
```json
{
  "prediction": "AI-generated",
  "confidence": 0.97
}
```

---

## Project Structure

```
AI-Text-Detection-Tool/
├── scripts/                 # Backend API & Dash launcher
│   ├── api_server.py
│   └── dashboard_app.py
├── notebooks/               # Jupyter analysis & Dash notebook
├── frontend/                # React single/batch analysis app
├── extension/               # Chrome extension source code
├── data/                    # Sample datasets for batch testing
├── models/                  # Trained model weights
├── requirements.txt         # Python dependencies
├── package.json             # Frontend dependencies
├── README.md                # This file
└── LICENSE                  # MIT license
```

---

## Contributing

1. Fork the repo  
2. Create a feature branch: `git checkout -b feature/awesome-new`
3. Commit your changes: `git commit -m "feat: add awesome feature"`
4. Push to your branch: `git push origin feature/awesome-new`
5. Open a Pull Request  

Please follow the [Contributing Guidelines](CONTRIBUTING.md) if available.

---

## Roadmap

- [ ] Add automated unit tests & CI via GitHub Actions  
- [ ] Containerize with Docker & Docker Compose  
- [ ] Publish Python package to PyPI  
- [ ] Enhance Chrome extension UI with dark mode toggle  

---

## Troubleshooting

- **API errors**? Check `scripts/api_server.py` logs for stack traces.  
- **Model loading issues**? Verify `MODEL_PATH` points to the correct `.pt` file.  
- **Port conflicts**? Change ports via `.env` or command-line flags.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

Michael Shpyl – [michael.shpyl@gmail.com](mailto:michael.shpyl@gmail.com)  
Project repo: https://github.com/MichaelShpyl/AI-Text-Detection-Tool

---

## Acknowledgments

- Built with ❤ using FastAPI, Plotly Dash, React, and LIME  
- Thanks to [DSP Sligo Business Improvement Team](https://www.gov.ie/dsp) for guidance  
