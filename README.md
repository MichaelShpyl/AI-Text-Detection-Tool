# AI-Text-Detection-Tool

**Detecting AI-Generated Text in Long-Form Content Using NLP Techniques**  
*A Study on Enhancing Academic Integrity and Improving AI Training*  

**Author:** Michael Shpyl  
**Supervisors:** Kevin Meehan, Mandy Douglas, Martin Robinson,  

---

## 🚀 Project Overview
This repository hosts a proof-of-concept pipeline for detecting AI-generated text in long-form content (news articles and academic essays). By combining advanced NLP preprocessing, statistical and hybrid detection methods, transformer-based architectures (BERT, RoBERTa, Longformer), and explainable AI techniques (LIME), the tool achieves high accuracy (≈97%) while providing transparent, provable evidence for its decisions.

Key contributions:
- Comprehensive preprocessing and feature engineering (tokenization, normalization, stopword removal, TF-IDF)  
- Custom dataset sampling from “All the News 2.0” with multi-stage AI paraphrasing via DeepSeek  
- Detection models: Random Forest, XGBoost, hybrid classifiers, and transformer-based detectors  
- Explainability module using LIME and contrastive semantic techniques  

---

## 📂 Repository Structure

```
AI-Text-Detection-Tool/
├── data/             # Raw and processed datasets (subset of All the News 2.0)
├── notebooks/        # Jupyter notebooks for preprocessing, model training, and analysis
├── scripts/          # ETL, data cleaning, training, evaluation scripts
├── extension/        # Browser extension for batch text classification
├── frontend/         # React dashboard for visualizing detection results
├── utils/            # Helper functions and modules
├── logs/             # Model checkpoints and training logs
├── EXAMINER_GUIDE.md # Instructions for examiners and deployment details
├── config.yaml       # Configuration for model parameters and training
├── requirements.txt  # Python dependencies
├── setup.py          # Package setup (editable install)
└── README.md         # This file
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MichaelShpyl/AI-Text-Detection-Tool.git
   cd AI-Text-Detection-Tool
   ```

2. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**  
   ```bash
   cd frontend
   npm install
   ```

---

## 💡 Usage

### 1. Data Preparation
```bash
python scripts/load_data.py \
  --input data/raw_dataset.csv \
  --output data/processed_dataset.csv
```

### 2. Model Training
```bash
python scripts/train.py --config config.yaml
```

### 3. API Server
```bash
uvicorn api.main:app --reload
```

### 4. Visualization Frontend
```bash
cd frontend
npm start
```

### 5. Browser Extension
- Load the `extension/` directory in Chrome/Edge developer mode to classify texts directly from any webpage.

---

## 🧪 Evaluation & Results
- Model accuracy: ~97% on held-out test set  
- Metrics: Precision, Recall, F1, ROC and PR curves (notebooks/analysis)  
- Explainability: LIME-based local and global explanations for classification decisions

---

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m 'feat: add new feature'`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request

Please follow the existing code style and include tests where applicable.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
