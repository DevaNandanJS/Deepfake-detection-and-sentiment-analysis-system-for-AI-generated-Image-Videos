import base64
import httpx
import json
import os

class ModerationEngine:
    """
    A service to evaluate the safety of an image using a local or remote Ollama instance.
    """
    def __init__(self):
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
        self.model_name = os.getenv("OLLAMA_MODEL", "llava:latest")
    
    def _encode_image_to_base64(self, file_path: str) -> str:
        """
        Reads a local image file and returns a valid base64 encoded string.
        """
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    async def evaluate_safety(self, file_path: str) -> dict:
        """
        Asynchronously evaluates the safety of an image by querying an Ollama instance.

        Args:
            file_path: The path to the image file to be evaluated.

        Returns:
            A dictionary containing the safety evaluation.
        """
        try:
            image_base64 = self._encode_image_to_base64(file_path)
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Is this image safe for work? Please respond with a JSON object like {\"safe\": true/false, \"reason\": \"...\"}.",
                        "images": [image_base64]
                    }
                ],
                "stream": False,
                "format": "json"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(self.ollama_api_url, json=payload, timeout=180.0)
                if response.status_code == 404:
                    return {"status": "error", "reason": f"Model '{self.model_name}' not found in Ollama."}
                response.raise_for_status()
                
            full_response = response.json()
            
            # Extract the content from the message
            if "message" in full_response and "content" in full_response["message"]:
                try:
                    # The content should be a JSON string because of "format": "json"
                    content_json = json.loads(full_response["message"]["content"])
                    return {
                        "status": "success",
                        "evaluation": content_json,
                        "raw_model_output": full_response["message"]["content"]
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success", 
                        "evaluation": {"safe": "unknown", "reason": full_response["message"]["content"]}
                    }
            
            return {"status": "error", "reason": "Unexpected response format from Ollama"}

        except httpx.ConnectError:
            return {"status": "error", "reason": "Ollama service is not running or unreachable."}
        except httpx.TimeoutException:
            return {"status": "error", "reason": "Ollama request timed out."}
        except Exception as e:
            print(f"An unexpected error occurred in ModerationEngine: {e}")
            return {"status": "error", "reason": str(e)}
