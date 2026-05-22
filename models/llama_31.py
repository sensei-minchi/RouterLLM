from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
class Llama:
    def __init__(self, model):
        api_key = os.getenv("GROQ_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model

    def response(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content