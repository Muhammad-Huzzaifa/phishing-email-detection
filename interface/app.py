import sys
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from src.predict import predict_email_alls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class PredictRequest(BaseModel):
    email_text: str = Field(..., min_length=1, description="Email content to classify")

class PredictResponse(BaseModel):
    predictions: dict[str, dict[str, int | float | str]]
    consensus: dict[str, float | int | str]

app = FastAPI(title="Phishing Email Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def _build_consensus(predictions: dict[str, dict[str, float | int | str]]) -> dict[str, float | int | str]:
    labels = [item["predicted_label"] for item in predictions.values()]
    phishing_votes = labels.count("Phishing")
    safe_votes = labels.count("Safe")
    final_label = "Phishing" if phishing_votes >= safe_votes else "Safe"

    avg_probability = sum(float(item["probability"]) for item in predictions.values()) / len(predictions)
    total_inference_ms = sum(float(item["inference_time_ms"]) for item in predictions.values())

    return {
        "final_label": final_label,
        "phishing_votes": phishing_votes,
        "safe_votes": safe_votes,
        "average_probability": round(avg_probability, 4),
        "total_inference_time_ms": round(total_inference_ms, 3),
    }

@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api")
def api_index() -> dict[str, Any]:
    return {
        "name": "Phishing Email Detector API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_post": "/api/predict",
            "docs": "/docs",
        },
        "note": "Use POST /api/predict with JSON body: {'email_text': '...'}",
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)

@app.get("/api/predict")
def predict_help() -> dict[str, str]:
    return {
        "message": "Method not allowed for prediction. Use POST /api/predict.",
        "example": "curl -X POST http://127.0.0.1:8000/api/predict -H 'Content-Type: application/json' -d '{\"email_text\": \"Sample email text\"}'",
    }

@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = payload.email_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Email text is required.")

    try:
        predictions = predict_email_alls(text)
        consensus = _build_consensus(predictions)
        return PredictResponse(predictions=predictions, consensus=consensus)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

if __name__ == "__main__":
    uvicorn.run("interface.app:app", host="0.0.0.0", port=8000, reload=True)
