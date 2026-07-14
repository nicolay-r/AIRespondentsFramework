import os

from openai import OpenAI


class OpenAIClient:

    _instance: "OpenAIClient | None" = None

    def __new__(cls) -> "OpenAIClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-32B",
        base_url: str = "https://api.studio.nebius.com/v1/",
    ) -> None:
        if self._initialized:
            return

        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY is not set. Add it to .env")

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._initialized = True

    def infer(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
