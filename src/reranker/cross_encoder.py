"""Cross-encoder reranking after hybrid RRF."""
from config import settings


class CrossEncoderReranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(settings.cross_encoder_model)

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        pairs = [[query, self._hit_text(hit)] for hit in hits]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, hits),
            key=lambda item: item[0],
            reverse=True,
        )
        top_hits = ranked[:settings.final_top_k]

        return [
            {**hit, "reranker_score": float(score)}
            for score, hit in top_hits
        ]

    @staticmethod
    def _hit_text(hit: dict) -> str:
        metadata = hit.get("metadata", {})
        return metadata.get("text") or metadata.get("chunk_text") or ""
