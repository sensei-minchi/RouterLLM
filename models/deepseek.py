import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class DeepSeek:
    """Wrapper for DeepSeek R1 Distill Llama 70B via the Groq API.

    DeepSeek R1 is a thinking model — its API responses include a
    <think>...</think> block containing internal chain-of-thought reasoning
    before the final answer.  We strip that block so that only the actual
    answer is stored and judged.
    """

    def __init__(self, model: str = "deepseek-r1-distill-llama-70b"):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> reasoning blocks from the response."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def response(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = completion.choices[0].message.content
            return self._strip_thinking(raw)
        except Exception as e:
            raise RuntimeError(f"DeepSeek API error: {e}") from e
