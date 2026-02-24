Act as a Lead Python Systems Architect. Your primary objective is to generate the foundational infrastructure, robust file tree, and dependency requirement configurations for a FastAPI application dedicated to multimodal synthetic media detection.
The system requires modularity, separating API routing from specific machine learning model inferences. The system will utilize Hugging Face 'transformers' for initial detection and 'ollama' via local HTTP requests for visual moderation.
Execute the following strict constraints:
Generate a valid bash script that constructs the complete directory structure.
The mandatory directory structure must include: 'app/api', 'app/core/config.py', 'app/services/deepfake_service.py', and 'app/services/moderation_service.py'.
Generate the exact content for a 'requirements.txt' file.
The 'requirements.txt' must include precisely: fastapi, uvicorn, python-multipart, transformers, torch, torchvision, httpx, opencv-python, and pillow.
Output the exact shell commands required to execute the bash script, initialize a Python virtual environment, and install the dependencies.
Provide the output strictly as functional code blocks. Avoid generating any conversational filler, qualitative bulleted lists, or explanations of the internal logic utilized to arrive at the solution.
