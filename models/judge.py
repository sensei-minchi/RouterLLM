from google import genai
from dotenv import load_dotenv
import os
import json

#loading the .env file
load_dotenv()


class GeminiJudge:
    def __init__(self, model="gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    #this will produce the response
    def judge(self, prompt, answer):
        judge_prompt = f"""
            You are a strict evaluation system for LLM responses.
            According to the user prompt, rate the LLM's response on a scale of 1-5

            User Prompt:
            {prompt}

            Answer:
            {answer}

            Return ONLY valid JSON in this format:
            {{
            "score": 1-5,
            "reason": "short explanation"
            }}

            Rules:
            - Be extremely strict
            - Prefer correctness over style
            - Penalize hallucinations heavily
            - Do NOT include markdown, text, or explanation outside JSON
            - Score can have decimal points upto 2 place - for eg 4.56
            """

        response = self.client.models.generate_content(
            model=self.model,
            contents=judge_prompt,
            generation_config={
                "temperature": 0
            }
        )

        text = response.text

        # Try parsing JSON safely
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "error": "invalid_json",
                "raw_output": text
            }