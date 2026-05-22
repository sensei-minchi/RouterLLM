from google import genai
from dotenv import load_dotenv
import os


load_dotenv()
class GeminiFlash:
    def __init__(self, model):
        api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    def response(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents = prompt
        )

        return response.text