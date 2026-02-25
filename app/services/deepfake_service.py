from transformers import pipeline
from PIL import Image, UnidentifiedImageError

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
            # Using the model as specified in the original code.
            self.pipe = pipeline("image-classification", model="ScuolaX/MT-deepfake-detector")
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
            A dictionary with the 'label', 'score', and 'all_predictions',
            or None if the analysis fails.
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
            
            best_prediction = max(result, key=lambda x: x['score'])
            
            # Normalize label to uppercase for robust comparison
            label = str(best_prediction['label']).upper()
            
            # Return the label, score, and the full set of predictions for debugging
            return {
                "label": label,
                "score": round(best_prediction['score'], 4),
                "all_predictions": result
            }
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
