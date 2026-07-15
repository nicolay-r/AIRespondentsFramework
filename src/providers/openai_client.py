import os

from openai import OpenAI


class OpenAIClient:

    def __init__(
        self,
        *,
        max_retries: int = 20,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        base_url: str = "https://api.studio.nebius.com/v1/",
    ) -> None:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY is not set. Add it to .env")

        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
        )

    def infer(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
