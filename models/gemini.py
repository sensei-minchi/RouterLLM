from google import genai
import os

class GeminiFlash:
    def __init__(self, model):
        api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(key=api_key)
        self.model = model
    
    def response(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents = prompt
        )

        return response.text