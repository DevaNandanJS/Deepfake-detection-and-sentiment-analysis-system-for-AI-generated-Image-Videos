import cv2
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
import os

class DeepfakeDetector:
    """
    A service to detect whether an image or video frame is real or a deepfake using a
    pre-trained Hugging Face model.
    """
    def __init__(self):
        """
        Initializes the DeepfakeDetector by loading the image classification
        pipeline from Hugging Face.
        """
        try:
            # Load the specified model for image classification
            self.pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")
        except Exception as e:
            # Handle potential errors during model loading (e.g., network issues)
            print(f"Error loading model: {e}")
            raise

    def detect(self, file_path: str) -> dict | None:
        """
        Analyzes an image or video from a given file path to determine if it is a
        deepfake.

        Args:
            file_path: The path to the file to be analyzed.

        Returns:
            A dictionary with the 'label' ('REAL' or 'FAKE') and a 'score',
            or None if the analysis fails.
        """
        image = None
        try:
            # Try to open as an image using Pillow
            image = Image.open(file_path)
        except (FileNotFoundError, UnidentifiedImageError):
            # If Pillow fails, try as a video using OpenCV
            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR (OpenCV) to RGB (Pillow)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                cap.release()
            except Exception as e:
                print(f"Error opening or processing file (image/video): {e}")
                return None

        if image is None:
            print(f"Could not extract any image from file: {file_path}")
            return None

        try:
            # Pass the image to the pipeline for classification
            result = self.pipe(image)

            # The result is a list of dictionaries. Find the one with the highest score.
            if not result:
                return None
            
            best_prediction = max(result, key=lambda x: x['score'])
            
            # Map labels to 'REAL' and 'FAKE'
            # The model 'prithivMLmods/Deep-Fake-Detector-v2-Model' uses 'Realism' and 'Deepfake'
            raw_label = best_prediction['label']
            if raw_label == 'Deepfake':
                label = 'FAKE'
            elif raw_label == 'Realism':
                label = 'REAL'
            else:
                label = raw_label # Fallback

            # Return the label and score of the most likely class
            return {
                "label": label,
                "score": round(best_prediction['score'], 4)
            }
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
