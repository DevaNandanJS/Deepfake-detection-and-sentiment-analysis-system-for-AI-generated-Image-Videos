import os
import tempfile
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.services.deepfake_service import DeepfakeDetector
from app.services.moderation_service import ModerationEngine
from app.services.sentiment_service import SentimentAnalyzer
from app.core.config import settings

# --- Data Models ---

class AnalysisResponse(BaseModel):
    is_synthetic: bool
    authenticity_score: float
    detected_label: str
    file_name: str
    content_type: str
    sentiment: Optional[Dict[str, Any]] = None
    moderation: Optional[Dict[str, Any]] = None
    debug_info: Optional[List[Dict[str, Any]]] = None

# --- Application Initialization ---

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="An API for detecting synthetic media, evaluating sentiment, and content safety.",
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load service models into memory on application startup
try:
    detector = DeepfakeDetector()
    sentiment_analyzer = SentimentAnalyzer()
except Exception as e:
    raise RuntimeError(f"Fatal: Could not load ML models. {e}")

moderator = ModerationEngine()

@app.post("/api/v1/analyze-media", response_model=AnalysisResponse)
async def analyze_media(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Accepts a media file and orchestrates the analysis pipeline:
    1. Authenticity Analysis
    2. Sentiment Analysis
    3. Safety Moderation (Conditional)
    """
    temp_file_path: Optional[str] = None
    try:
        # Save upload to temporary file
        suffix = os.path.splitext(file.filename)[1] if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # --- Stage 1: Authenticity Analysis ---
        authenticity_result = detector.detect(temp_file_path)
        if not authenticity_result:
            raise HTTPException(status_code=500, detail="Authenticity analysis failed.")

        label = authenticity_result.get("label", "").upper()
        score = authenticity_result.get("score", 0.0)
        
        is_synthetic = (label in settings.SYNTHETIC_LABELS and score >= settings.CONFIDENCE_THRESHOLD)

        # --- Stage 2: Sentiment Analysis ---
        sentiment_result = None
        if file.content_type and file.content_type.startswith("image/"):
            sentiment_result = sentiment_analyzer.analyze(temp_file_path)

        # Build initial response
        response_data = {
            "is_synthetic": is_synthetic,
            "authenticity_score": score,
            "detected_label": label,
            "file_name": file.filename or "unknown",
            "content_type": file.content_type or "application/octet-stream",
            "sentiment": sentiment_result,
            "debug_info": authenticity_result.get("all_predictions")
        }

        # --- Stage 3: Conditional Safety Moderation ---
        if is_synthetic:
            if file.content_type and file.content_type.startswith("image/"):
                response_data["moderation"] = await moderator.evaluate_safety(temp_file_path)
            elif file.content_type and file.content_type.startswith("video/"):
                response_data["moderation"] = {
                    "status": "skipped",
                    "reason": "Video moderation requires frame extraction (Coming Soon)."
                }

        return AnalysisResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Cleanup error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
