from transformers import pipeline
from PIL import Image, UnidentifiedImageError
from app.core.config import settings

class DeepfakeDetector:
    """
    A service to detect whether an image is real or a deepfake using a
    pre-trained Hugging Face model.
    """
    def __init__(self):
        """
        Initializes the DeepfakeDetector by loading the image classification
        pipeline from Hugging Face.
        """
        try:
            # Load the specified model for image classification
            self.pipe = pipeline("image-classification", model=settings.DEEPFAKE_MODEL)
        except Exception as e:
            # Handle potential errors during model loading (e.g., network issues)
            print(f"Error loading model: {e}")
            raise

    def detect(self, file_path: str) -> dict | None:
        """
        Analyzes an image from a given file path to determine if it is a
        deepfake.

        Args:
            file_path: The path to the image file to be analyzed.

        Returns:
            A dictionary with 'real_score', 'fake_score', 'best_label', 
            'best_score', and 'all_predictions', or None if the analysis fails.
        """
        try:
            # Open the image using Pillow and convert to RGB to ensure compatibility
            image = Image.open(file_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error opening or processing image file: {e}")
            return None

        try:
            # Pass the image to the pipeline for classification
            result = self.pipe(image)

            # The result is a list of dictionaries.
            if not result:
                return None
            
            # Create a mapping of uppercase labels to scores for easier access
            predictions = {str(p['label']).upper(): p['score'] for p in result}
            
            # Identify the best overall prediction
            best_prediction = max(result, key=lambda x: x['score'])
            best_label_raw = str(best_prediction['label']).upper()
            best_score = best_prediction['score']

            # Determine explicit scores for REAL and FAKE (or similar)
            # 0: real, 1: fake is the most common mapping for deepfake binary classifiers
            # Adding REALISM and DEEPFAKE for newer model compatibility
            real_score = predictions.get('REAL', predictions.get('REALISM', predictions.get('LABEL_0', 0.0)))
            fake_score = predictions.get('FAKE', predictions.get('DEEPFAKE', predictions.get('LABEL_1', 0.0)))

            # If we couldn't find explicit REAL/FAKE or LABEL_0/1, try logic-based inference
            if all(k not in predictions for k in ['REAL', 'REALISM', 'LABEL_0']):
                # If we have a known synthetic label, real is 1 - synthetic
                for syn_label in ['FAKE', 'DEEPFAKE', 'SYNTHETIC', 'GENERATED', 'LABEL_1']:
                    if syn_label in predictions:
                        real_score = 1.0 - predictions[syn_label]
                        break
            
            if all(k not in predictions for k in ['FAKE', 'DEEPFAKE', 'LABEL_1']):
                fake_score = 1.0 - real_score

            return {
                "best_label": best_label_raw,
                "best_score": round(best_score, 4),
                "real_score": round(real_score, 4),
                "fake_score": round(fake_score, 4),
                "all_predictions": result
            }
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
