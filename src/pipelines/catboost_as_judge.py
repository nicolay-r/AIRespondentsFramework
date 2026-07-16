from src.pipelines.base import PipelineItem
from src.pipelines.catboost_statements import CatBoostStatementsPipeline


class CatBoostAsJudgePipeline(CatBoostStatementsPipeline):

    def _catboost_confidence(
        self,
        item: PipelineItem,
        catboost_prediction: str | None,
    ) -> int | None:
        if (
            catboost_prediction is None
            or item.question_id not in self._recommender.models
        ):
            return None

        probabilities = self._recommender.predict_proba(
            self._respondent_frame(item),
            item.question_id,
        )
        if catboost_prediction in probabilities.columns:
            return round(probabilities[catboost_prediction].iloc[0] * 100)
        return round(probabilities.max(axis=1).iloc[0] * 100)

    def build_explain_prompt(self, item: PipelineItem) -> str:
        ranked_entries = self._ranked_entries(item)
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
            "",
            "Most relevant background (most to least relevant):",
            *self._profile_lines(item, ranked_entries),
            "",
            f"Question: {item.question}",
            "Explain your reasoning and state your answer in a single paragraph.",
            f"End with exactly one of: {', '.join(item.labels)}",
        ]
        return "\n".join(lines)

    def build_judge_prompt(
        self,
        item: PipelineItem,
        explanation: str,
        *,
        catboost_prediction: str | None,
        catboost_confidence: int | None,
    ) -> str:
        lines = [
            "You are judging which survey answer is most appropriate.",
            "You must choose between two sources:",
            "1. The respondent analysis below",
            "2. The CatBoost model prediction below",
            "",
            f"Question: {item.question}",
            f"Allowed answers: {', '.join(item.labels)}",
            "",
            "Respondent analysis:",
            explanation.strip(),
        ]

        if catboost_prediction is not None:
            if catboost_confidence is not None:
                lines.extend(
                    [
                        "",
                        "CatBoost model prediction:",
                        f"{catboost_prediction} ({catboost_confidence}%)",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "CatBoost model prediction:",
                        catboost_prediction,
                    ]
                )

        lines.extend(
            [
                "",
                "Weigh the two sources above and pick one allowed answer.",
                #f"Answer with exactly one of: {', '.join(item.labels)}",
                "Reply with the label only.",
            ]
        )
        return "\n".join(lines)

    def build_prompt(self, item: PipelineItem) -> str:
        return self.build_explain_prompt(item)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        ordered = self._ordered_entries(item)
        catboost_prediction = self._catboost_prediction(item)
        catboost_confidence = self._catboost_confidence(item, catboost_prediction)

        explanation = self._client.infer(self.build_explain_prompt(item))

        judge_prompt = self.build_judge_prompt(
            item,
            explanation,
            catboost_prediction=catboost_prediction,
            catboost_confidence=catboost_confidence,
        )

        output = self._client.infer(judge_prompt)
        print("judge_prompt: ", judge_prompt + "\n\n" + output + "\n\n================") 

        return {
            "output": output,
            "explanation": explanation,
            "features": [entry.code for entry in ordered],
            "catboost_prediction": catboost_prediction,
            "catboost_confidence": catboost_confidence,
        }
