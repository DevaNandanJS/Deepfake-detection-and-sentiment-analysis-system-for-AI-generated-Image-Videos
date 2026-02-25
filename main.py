import os
import tempfile
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from app.services.deepfake_service import DeepfakeDetector
from app.services.moderation_service import ModerationEngine

# --- Global Instantiation ---
# Initialize the FastAPI application
app = FastAPI(
    title="Multimodal Media Analysis System",
    description="An API for detecting synthetic media and evaluating content safety.",
    version="1.0.0",
)

# Load service models into memory on application startup
try:
    detector = DeepfakeDetector()
except Exception as e:
    raise RuntimeError(f"Fatal: Could not load DeepfakeDetector model. {e}")

moderator = ModerationEngine()


@app.post("/api/v1/analyze-media", response_model=Dict[str, Any])
async def analyze_media(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts a media file, orchestrates a sequential analysis pipeline,
    and returns a unified JSON response.

    Pipeline:
    1.  **Authenticity Analysis**: Determines if the media is synthetic.
    2.  **Conditional Safety Moderation**: If media is a synthetic image,
        it's evaluated against safety taxonomies.
    """
    temp_file_path: Optional[str] = None
    try:
        # --- Secure Temporary File Handling ---
        # Create a named temporary file to securely store the upload on disk.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # --- Step One: Authenticity Analysis ---
        # Pass the file path to the deepfake detector.
        authenticity_result = detector.detect(temp_file_path)
        if not authenticity_result:
            raise HTTPException(status_code=500, detail="Authenticity analysis failed.")

        # Extract label and score
        label = authenticity_result.get("label", "").upper()
        score = authenticity_result.get("score", 0.0)
        
        # Define keywords that indicate synthetic/fake media
        SYNTHETIC_LABELS = {"FAKE", "SYNTHETIC", "GENERATED", "AI_GENERATED", "DEEPFAKE"}

        # Confidence Threshold: adjust this value (0.0 to 1.0) to balance 
        # sensitivity and false positives. High values (e.g. 0.8) are more strict.
        CONFIDENCE_THRESHOLD = 0.8
        
        is_synthetic = (label in SYNTHETIC_LABELS and score >= CONFIDENCE_THRESHOLD)

        # Structure the initial response
        final_response = {
            "is_synthetic": is_synthetic,
            "authenticity_score": score,
            "detected_label": label,
            "file_name": file.filename,
            "content_type": file.content_type,
            "debug_info": authenticity_result.get("all_predictions")  # Helpful for debugging false positives
        }

        # --- Conditional Logic Gate: Safety Moderation ---
        if is_synthetic:
            # If media is a synthetic image, proceed to moderation.
            if file.content_type and file.content_type.startswith("image/"):
                safety_result = await moderator.evaluate_safety(temp_file_path)
                final_response["moderation"] = safety_result
            # If media is a video, add a placeholder for future implementation.
            elif file.content_type and file.content_type.startswith("video/"):
                # TODO: Implement video frame extraction logic here.
                # For now, we add a placeholder note to the response.
                final_response["moderation"] = {
                    "status": "skipped",
                    "reason": "Video moderation requires frame extraction and is not yet implemented."
                }

        return final_response

    except Exception as e:
        # Catch-all for unexpected errors during the process.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        # --- Guaranteed Cleanup ---
        # Ensure the temporary file is deleted after the request is complete.
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
