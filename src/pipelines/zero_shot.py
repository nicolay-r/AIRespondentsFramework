from src.pipelines.base import Pipeline, PipelineItem
from src.providers.openai_client import OpenAIClient


class ZeroShotPipeline(Pipeline):
    
    def __init__(self, client: OpenAIClient) -> None:
        self._client = client

    def build_prompt(self, item: PipelineItem) -> str:
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
        ]
        for question, answer in item.history.items():
            lines.append(f"- {question}: {answer if answer is not None else 'Unknown'}")

        lines.extend(
            [
                "",
                f"Question: {item.question}",
                f"Answer with exactly one of: {', '.join(item.labels)}",
            ]
        )
        return "\n".join(lines)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        return {
            "output": self._client.infer(self.build_prompt(item)),
            "features": item.features,
        }
