# Phishing Email Detection

A complete NLP pipeline for phishing email classification using four models:
- Logistic Regression (TF-IDF)
- Feedforward Neural Network (FNN)
- Simple RNN
- LSTM

The project includes:
- data download and preprocessing
- model training/evaluation notebooks
- a FastAPI backend
- a responsive Neo-Brutalist web interface for live inference

## Project Structure

```text
phishing-email-detection/
├── data/
│   ├── raw/
│   │   └── phishing_email.csv
│   └── processed/
│       ├── X_*_tfidf.npz
│       ├── X_*_seq.npy
│       ├── y_*.csv
│       └── *_pred.csv
├── interface/
│   ├── app.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── models/
│   ├── lr.pkl
│   ├── fnn.keras
│   ├── rnn.keras
│   ├── lstm.keras
│   ├── tfidf_vectorizer.joblib
│   └── tokenizer.pkl
├── notebooks/
│   ├── preprocess.ipynb
│   ├── models.ipynb
│   └── evaluate.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── *_confusion_matrix.png
│   ├── *_loss_curves.png
│   ├── roc_curve.png
│   └── training_time_comparison.png
├── src/
│   ├── download.py
│   ├── preprocess.py
│   └── predict.py
└── requirements.txt
```

## End-to-End Process

1. Download dataset
- Script: `src/download.py`
- Source: Kaggle dataset `naserabdullahalam/phishing-email-dataset`

2. Preprocess data
- Notebook: `notebooks/preprocess.ipynb`
- Creates train/val/test split.
- Builds:
  - TF-IDF features for Logistic Regression
  - tokenized + padded sequences for neural models
- Saves processed arrays and labels to `data/processed/`.
- Saves `tfidf_vectorizer.joblib` and `tokenizer.pkl` to `models/`.

3. Train models
- Notebook: `notebooks/models.ipynb`
- Trains LR, FNN, RNN, LSTM.
- Saves trained model artifacts to `models/`.
- Saves prediction CSVs and training plots.

4. Evaluate models
- Notebook: `notebooks/evaluate.ipynb`
- Produces confusion matrices, ROC curve, and comparison CSV.

5. Serve inference API + web app
- Backend: `interface/app.py` (FastAPI)
- Inference engine: `src/predict.py`
- Shared text processing: `src/preprocess.py`
- Frontend: `interface/static/*`

## Methods

### Text Preprocessing
Implemented in `src/preprocess.py`:
- lowercasing
- whitespace normalization (`\s+` -> single space)
- input validation (non-empty text)

### Feature Pipelines
- Logistic Regression: TF-IDF vectorization
- FNN/RNN/LSTM: tokenizer + padded sequence (`maxlen=500`)

### Prediction Flow
Implemented in `src/predict.py`:
- load all artifacts once (cached predictor)
- run all four models on the same input
- return for each model:
  - class (`0/1`)
  - label (`Safe/Phishing`)
  - probability
  - inference time in milliseconds

### API Layer
Implemented in `interface/app.py`:
- `GET /` serves web interface
- `GET /api` API overview
- `GET /health` health check
- `POST /api/predict` run inference
- `GET /api/predict` helper message for method usage

## Results

From `results/model_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| LR   | 0.9912 | 0.9884 | 0.9947 | 0.9915 |
| FNN  | 0.9884 | 0.9855 | 0.9923 | 0.9889 |
| RNN  | 0.4843 | 0.7667 | 0.0160 | 0.0314 |
| LSTM | 0.9896 | 0.9895 | 0.9905 | 0.9900 |

Summary:
- LR, FNN, and LSTM perform strongly.
- RNN underperforms significantly in this setup.

## Running Instructions

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) (Optional) Download dataset

```bash
python src/download.py
```

## 3) (Optional) Rebuild preprocessing/training/evaluation
Run notebooks in this order:
1. `notebooks/preprocess.ipynb`
2. `notebooks/models.ipynb`
3. `notebooks/evaluate.ipynb`

## 4) Start API + Web App

```bash
uvicorn interface.app:app --reload
```

Open:
- Web app: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 5) Test Prediction via cURL

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Your account is suspended. Verify now."}'
```

## Notes

- `scikit-learn` version mismatch warnings can appear when loading older pickled models; retraining with the current environment removes this warning.
- First inference can be slower due to TensorFlow model initialization.

## Tech Stack

- Python, NumPy, Pandas
- scikit-learn
- TensorFlow/Keras
- FastAPI + Uvicorn
- HTML/CSS/JS frontend
