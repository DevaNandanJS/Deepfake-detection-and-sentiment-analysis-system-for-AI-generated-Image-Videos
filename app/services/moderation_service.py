import base64
import httpx
import json
from app.core.config import settings

class ModerationEngine:
    """
    A service to evaluate the safety of an image using a local Ollama instance.
    """
    
    def _encode_image_to_base64(self, file_path: str) -> str:
        """
        Reads a local image file and returns a valid base64 encoded string,
        stripping any newline characters.
        """
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string.replace("\n", "")

    async def evaluate_safety(self, file_path: str) -> dict:
        """
        Asynchronously evaluates the safety of an image by querying a local Ollama
        instance.

        Args:
            file_path: The path to the image file to be evaluated.

        Returns:
            A dictionary containing the safety evaluation from the Ollama service,
            or a standardized fallback dictionary in case of an error.
        """
        try:
            image_base64 = self._encode_image_to_base64(file_path)
            
            payload = {
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Is this image safe for work? Please respond with a single word: 'safe' or 'unsafe'.",
                        "images": [image_base64]
                    }
                ],
                "stream": False,
                "format": "json"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(settings.OLLAMA_URL, json=payload, timeout=30.0)
                response.raise_for_status()
                
            return json.loads(response.text)

        except httpx.RequestError as e:
            print(f"Error querying Ollama: {e}")
            return {"status": "error", "reason": "Ollama service unreachable"}
        except json.JSONDecodeError as e:
            print(f"Error parsing Ollama response: {e}")
            return {"status": "error", "reason": "Invalid JSON response from Ollama"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"status": "error", "reason": "An unexpected error occurred"}
