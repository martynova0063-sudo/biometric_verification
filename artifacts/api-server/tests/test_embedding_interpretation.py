import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi import HTTPException

from app import main
from app.embedding_interpretation import (
    EmbeddingInterpretationNotConfigured,
    build_interpretation_metrics,
    generate_embedding_interpretation,
)
from app.schemas import EmbeddingInterpretationRequest


def sample_comparison():
    return {
        "status": "available",
        "method": "resnet50_arcface",
        "reference_vectors": 6,
        "video_vectors": 5,
        "dimensions": 512,
        "mean": {
            "reference": 0.0036,
            "video": 0.0042,
            "delta": 0.0006,
            "centroid_cosine_similarity": 0.701,
        },
        "variance": {"reference": 0.0019, "video": 0.0019, "ratio": 0.9978},
        "outliers": {"count": 0, "total": 5, "threshold": 0.39},
        "distribution": {
            "bin_centers": [-0.01, 0.01],
            "reference_counts": [10, 10],
            "video_counts": [10, 10],
        },
        "employee_ref": "0001",
        "session_id": "private-session-id",
        "raw_vectors": [[0.1, 0.2]],
    }


class EmbeddingInterpretationTests(unittest.IsolatedAsyncioTestCase):
    def test_llm_payload_contains_aggregates_only(self):
        comparison = sample_comparison()

        metrics = build_interpretation_metrics(comparison)

        self.assertEqual(metrics["centroid_cosine_similarity"], 0.701)
        self.assertEqual(metrics["normalized_histogram_overlap"], 1.0)
        self.assertNotIn("employee_ref", metrics)
        self.assertNotIn("session_id", metrics)
        self.assertNotIn("raw_vectors", metrics)

    async def test_interpretation_uses_json_output_and_persists_metadata(self):
        draft = {
            "verdict": "borderline",
            "summary": "Центроидное сходство находится у нижней границы ориентиров.",
            "findings": ["Разброс близок к эталонному."],
            "recommendations": ["Проверьте калибровку порога для этой модели."],
        }
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(draft, ensure_ascii=False))
                )
            ]
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.chat.completions.create = AsyncMock(return_value=completion)

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-key"}),
            patch("app.embedding_interpretation.AsyncOpenAI", return_value=client),
        ):
            result = await generate_embedding_interpretation(sample_comparison())

        self.assertEqual(result["verdict"], "borderline")
        self.assertEqual(result["model"], "gpt-5.4-mini")
        self.assertEqual(result["analysis_version"], 1)
        self.assertEqual(result["disclaimer"].startswith("Исследовательское"), True)
        request_messages = client.chat.completions.create.await_args.kwargs["messages"]
        self.assertNotIn("private-session-id", request_messages[1]["content"])
        self.assertNotIn("raw_vectors", request_messages[1]["content"])

    async def test_missing_api_key_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EmbeddingInterpretationNotConfigured):
                await generate_embedding_interpretation(sample_comparison())

    async def test_endpoint_saves_result_and_reuses_cached_interpretation(self):
        session_id = uuid4()
        session = SimpleNamespace(
            details={"embedding_comparison": sample_comparison()}
        )
        db = Mock()
        db.get.return_value = session
        generated = {
            "analysis_version": 1,
            "verdict": "borderline",
            "summary": "Пограничное сходство.",
            "findings": ["Распределения требуют проверки."],
            "recommendations": ["Проверьте кадрирование."],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "gpt-5.4-mini",
            "disclaimer": "Исследовательское объяснение.",
        }
        generator = AsyncMock(return_value=generated)
        payload = EmbeddingInterpretationRequest(refresh=False)

        with patch.object(main, "generate_embedding_interpretation", generator):
            result = await main.create_embedding_interpretation(
                session_id, payload, db
            )
            cached_result = await main.create_embedding_interpretation(
                session_id, payload, db
            )

        self.assertEqual(result, generated)
        self.assertEqual(cached_result, generated)
        self.assertEqual(session.details["embedding_interpretation"], generated)
        self.assertEqual(generator.await_count, 1)
        db.commit.assert_called_once()

    async def test_endpoint_rejects_sessions_without_comparison_data(self):
        db = Mock()
        db.get.return_value = SimpleNamespace(details={})

        with self.assertRaises(HTTPException) as raised:
            await main.create_embedding_interpretation(
                uuid4(), EmbeddingInterpretationRequest(), db
            )

        self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
