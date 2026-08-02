import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class LLMClient:
    def __init__(self, model="google/gemma-4-26b-a4b-it:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model
        self.history = []

    def ask(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                timeout=15,
            )
            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def __str__(self):
        return f"LLMClient(model={self.model})"
