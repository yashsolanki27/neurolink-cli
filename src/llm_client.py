import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class LLMClient:
    def __init__(self, model="nvidia/nemotron-3-ultra-550b-a55b:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model

    def ask(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=15,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def __str__(self):
        return f"LLMClient(model={self.model})"
