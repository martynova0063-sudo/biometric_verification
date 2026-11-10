import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError


logger = logging.getLogger(__name__)
OPENAI_MODEL = "gpt-5.4-mini"
INTERPRETATION_VERSION = 1
INTERPRETATION_DISCLAIMER = (
    "Исследовательское объяснение агрегированных метрик. "
    "Оно не подтверждает и не опровергает личность и не меняет решение системы."
)


class EmbeddingInterpretationNotConfigured(Exception):
    pass


class EmbeddingInterpretationGenerationError(Exception):
    pass


class _InterpretationDraft(BaseModel):
    verdict: Literal["consistent", "borderline", "discordant", "insufficient_data"]
    summary: str = Field(min_length=1, max_length=500)
    findings: list[str] = Field(min_length=1, max_length=4)
    recommendations: list[str] = Field(min_length=1, max_length=4)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return round(numeric, 8) if math.isfinite(numeric) else None


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _histogram_overlap(distribution: Any) -> float | None:
    if not isinstance(distribution, dict):
        return None
    reference = distribution.get("reference_counts")
    video = distribution.get("video_counts")
    if not isinstance(reference, list) or not isinstance(video, list) or len(reference) != len(video):
        return None

    reference_counts = [_count(value) for value in reference]
    video_counts = [_count(value) for value in video]
    reference_total = sum(reference_counts)
    video_total = sum(video_counts)
    if not reference_total or not video_total:
        return None

    overlap = sum(
        min(reference_value / reference_total, video_value / video_total)
        for reference_value, video_value in zip(reference_counts, video_counts)
    )
    return round(overlap, 4)


def build_interpretation_metrics(comparison: dict[str, Any]) -> dict[str, Any]:
    """Keep the LLM input limited to summary statistics, never vectors or identifiers."""
    mean = comparison.get("mean") if isinstance(comparison.get("mean"), dict) else {}
    variance = comparison.get("variance") if isinstance(comparison.get("variance"), dict) else {}
    outliers = comparison.get("outliers") if isinstance(comparison.get("outliers"), dict) else {}
    method = comparison.get("method")

    return {
        "method": method[:80] if isinstance(method, str) else "не указан",
        "reference_vectors": _count(comparison.get("reference_vectors")),
        "video_vectors": _count(comparison.get("video_vectors")),
        "dimensions": _count(comparison.get("dimensions")),
        "mean_components": {
            "reference": _finite_number(mean.get("reference")),
            "video": _finite_number(mean.get("video")),
            "video_minus_reference": _finite_number(mean.get("delta")),
        },
        "variance_ratio_video_to_reference": _finite_number(variance.get("ratio")),
        "centroid_cosine_similarity": _finite_number(
            mean.get("centroid_cosine_similarity")
        ),
        "outliers": {
            "count": _count(outliers.get("count")),
            "total_video_vectors": _count(outliers.get("total")),
            "threshold": _finite_number(outliers.get("threshold")),
            "calculated": isinstance(outliers.get("count"), int)
            and not isinstance(outliers.get("count"), bool),
        },
        "normalized_histogram_overlap": _histogram_overlap(
            comparison.get("distribution")
        ),
    }


async def generate_embedding_interpretation(
    comparison: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingInterpretationNotConfigured

    metrics = build_interpretation_metrics(comparison)
    system_prompt = (
        "Ты — аналитик качества эмбеддингов, а не система распознавания личности. "
        "Интерпретируй только переданные агрегированные метрики и не додумывай факты. "
        "Ответь на русском языке и верни JSON с полями: verdict, summary, findings, recommendations. "
        "Допустимые verdict: consistent (агрегаты в целом согласуются), borderline "
        "(результат пограничный или метрики противоречат друг другу), discordant "
        "(есть выраженное техническое расхождение), insufficient_data (данных недостаточно). "
        "Оценивай центроидное косинусное сходство вместе с перекрытием распределений, "
        "числом векторов и выбросами. Малое число эталонных векторов ограничивает надёжность; "
        "если выбросы не рассчитаны, так и укажи. Значение около 0.70 из примеров проекта — "
        "только ориентир для обсуждения, не универсальный порог; порог зависит от модели и "
        "калибровки FAR/FRR. Никогда не утверждай, что личность подтверждена или опровергнута, "
        "и не предлагай автоматически менять порог, эталон или индекс. При расхождении предложи "
        "проверить кадры и предобработку (кадрирование, BGR/RGB, нормализацию); вывод должен "
        "оставаться технической рекомендацией для специалиста. summary — максимум 2 предложения; "
        "findings — 1–4 коротких наблюдения; recommendations — 1–4 конкретные проверки."
    )

    try:
        async with AsyncOpenAI(api_key=api_key, timeout=35.0) as client:
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(metrics, ensure_ascii=False),
                    },
                ],
            )
    except Exception as error:
        logger.warning(
            "OpenAI embedding interpretation request failed (%s).",
            type(error).__name__,
        )
        raise EmbeddingInterpretationGenerationError from error

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise EmbeddingInterpretationGenerationError("empty_response")

    try:
        draft = _InterpretationDraft.model_validate_json(content)
    except (ValidationError, ValueError, TypeError) as error:
        logger.warning(
            "OpenAI embedding interpretation returned invalid JSON (%s).",
            type(error).__name__,
        )
        raise EmbeddingInterpretationGenerationError("invalid_response") from error

    return {
        **draft.model_dump(),
        "analysis_version": INTERPRETATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": OPENAI_MODEL,
        "disclaimer": INTERPRETATION_DISCLAIMER,
    }
