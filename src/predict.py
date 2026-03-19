from dataclasses import dataclass
from pathlib import Path
import pickle
import joblib
import numpy as np
import tensorflow as tf
from time import perf_counter
from src.preprocess import (
    decode_prediction,
    tfidf_vectorize,
    tokenize_and_pad,
    validate_input,
)

@dataclass
class ModelPrediction:
    model_name: str
    predicted_class: int
    predicted_label: str
    probability: float
    inference_time_ms: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model_name,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "probability": self.probability,
            "inference_time_ms": self.inference_time_ms,
        }

def _sigmoid(x: np.ndarray) -> np.ndarray:
	return 1.0 / (1.0 + np.exp(-x))

class PhishingPredictor:

	def __init__(self, maxlen: int = 500):
		self.models_dir = Path(__file__).resolve().parent.parent / "models"
		self.maxlen = maxlen

		self.lr = None
		self.fnn = None
		self.rnn = None
		self.lstm = None
		self.tfidf_vectorizer = None
		self.tokenizer = None

		self._load_artifacts()

	def _load_artifacts(self) -> None:
		with open(self.models_dir / "lr.pkl", "rb") as f:
			self.lr = pickle.load(f)

		self.fnn = tf.keras.models.load_model(self.models_dir / "fnn.keras")
		self.rnn = tf.keras.models.load_model(self.models_dir / "rnn.keras")
		self.lstm = tf.keras.models.load_model(self.models_dir / "lstm.keras")
		self.tfidf_vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")

		with open(self.models_dir / "tokenizer.pkl", "rb") as f:
			self.tokenizer = pickle.load(f)

	def predict_all(self, text: str) -> dict[str, dict[str, float | int | str]]:
		cleaned = validate_input(text)

		tfidf_input = tfidf_vectorize(cleaned, self.tfidf_vectorizer)
		seq_input = tokenize_and_pad(
			cleaned,
			tokenizer=self.tokenizer,
			maxlen=self.maxlen,
		)

		outputs = {
			"Logistic Regression": self._predict_lr(tfidf_input),
			"FNN": self._predict_nn("FNN", self.fnn, seq_input),
			"RNN": self._predict_nn("RNN", self.rnn, seq_input),
			"LSTM": self._predict_nn("LSTM", self.lstm, seq_input),
		}
		return {name: result.as_dict() for name, result in outputs.items()}

	def _predict_lr(self, tfidf_input: np.ndarray) -> ModelPrediction:
		start = perf_counter()

		if hasattr(self.lr, "predict_proba"):
			probability = float(self.lr.predict_proba(tfidf_input)[0, 1])
		else:
			probability = float(self.lr.predict(tfidf_input)[0])

		inference_time_ms = (perf_counter() - start) * 1000
		predicted_class = int(probability >= 0.5)

		return ModelPrediction(
			model_name="Logistic Regression",
			predicted_class=predicted_class,
			predicted_label=decode_prediction(predicted_class),
			probability=probability,
			inference_time_ms=inference_time_ms,
		)

	def _predict_nn(self, model_name: str, model: tf.keras.Model, seq_input: np.ndarray) -> ModelPrediction:
		start = perf_counter()
		logits = model.predict(seq_input, verbose=0)
		probability = float(_sigmoid(np.array(logits))[0, 0])
		inference_time_ms = (perf_counter() - start) * 1000
		predicted_class = int(probability >= 0.5)

		return ModelPrediction(
			model_name=model_name,
			predicted_class=predicted_class,
			predicted_label=decode_prediction(predicted_class),
			probability=probability,
			inference_time_ms=inference_time_ms,
		)

_PREDICTOR: PhishingPredictor | None = None

def get_predictor() -> PhishingPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = PhishingPredictor()
    return _PREDICTOR


def predict_email_alls(text: str) -> dict[str, dict[str, float | int | str]]:
    return get_predictor().predict_all(text)
