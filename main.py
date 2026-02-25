import os
import tempfile
import mimetypes
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
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


@app.get("/")
async def serve_frontend():
    """Serves the frontend application."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend file not found.")


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
        suffix = os.path.splitext(file.filename)[1] if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Determine content type if not provided or vague
        content_type = file.content_type
        if not content_type or content_type == "application/octet-stream":
            content_type, _ = mimetypes.guess_type(file.filename or "")

        # --- Step One: Authenticity Analysis ---
        # Pass the file path to the deepfake detector.
        authenticity_result = detector.detect(temp_file_path)
        if not authenticity_result:
            raise HTTPException(status_code=500, detail="Authenticity analysis failed.")

        # Structure the initial response
        is_synthetic = authenticity_result.get("label") == "FAKE"
        final_response = {
            "is_synthetic": is_synthetic,
            "authenticity_score": authenticity_result.get("score"),
            "file_name": file.filename,
            "content_type": content_type,
        }

        # --- Conditional Logic Gate: Safety Moderation ---
        if is_synthetic:
            # If media is a synthetic image, proceed to moderation.
            if content_type and content_type.startswith("image/"):
                safety_result = await moderator.evaluate_safety(temp_file_path)
                final_response["moderation"] = safety_result
            # If media is a video, add a placeholder for future implementation.
            elif content_type and content_type.startswith("video/"):
                # For now, we still add a placeholder note, but we could extend this.
                final_response["moderation"] = {
                    "status": "skipped",
                    "reason": "Video moderation requires full frame analysis and is not yet implemented."
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
