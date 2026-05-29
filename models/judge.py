from google import genai
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json
from pathlib import Path

# Load the .env file from the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class GeminiJudge:
    def __init__(self, model="gemini-2.5-flash"):
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
            contents=judge_prompt
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


class LlamaJudge:
    def __init__(self, model="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model

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

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": judge_prompt}
                ],
                response_format={"type": "json_object"}
            )
            text = completion.choices[0].message.content
        except Exception as e:
            return {
                "error": f"API Error: {str(e)}",
                "raw_output": ""
            }

        # Try parsing JSON safely
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Let's try cleaning the text if it includes markdown blocks
            cleaned_text = text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                return {
                    "error": "invalid_json",
                    "raw_output": text
                }


class DeepSeekJudge:
    """LLM-as-a-Judge using DeepSeek R1 Distill Llama 70B via the Groq API.

    DeepSeek R1 is a thinking model: its output wraps internal chain-of-thought
    in <think>...</think> tags before the final answer.  We strip those tags
    so only the JSON verdict is parsed.
    """

    def __init__(self, model: str = "qwen/qwen3-32b"):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> reasoning blocks from the model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def judge(self, prompt: str, answer: str) -> dict:
        judge_prompt = f"""
You are a strict evaluation system for LLM responses.
According to the user prompt, rate the LLM's response on a scale of 1-5.

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
- Score can have decimal points up to 2 places - e.g. 4.56
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            raw = completion.choices[0].message.content
            text = self._strip_thinking(raw)
        except Exception as e:
            return {"error": f"API Error: {str(e)}", "raw_output": ""}

        # Try parsing JSON directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Clean markdown fences if present
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"error": "invalid_json", "raw_output": text}