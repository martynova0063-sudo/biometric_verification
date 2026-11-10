from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import ActivityItem, HealthResult, ReviewTask, VerificationSession
from .schemas import HealthCalculationInput


DISCLAIMER = "Исследовательский результат. Не является медицинским диагнозом."

async def inspect_upload(upload: UploadFile, media_kind: str, max_bytes: int) -> dict[str, Any]:
    extension = (upload.filename or "").rsplit(".", 1)[-1].lower() if "." in (upload.filename or "") else ""
    allowed = {
        "photo": {
            "jpg": {"image/jpeg"},
            "jpeg": {"image/jpeg"},
            "png": {"image/png"},
            "webp": {"image/webp"},
        },
        "video": {
            "mp4": {"video/mp4"},
            "m4v": {"video/mp4", "video/x-m4v"},
            "mov": {"video/quicktime"},
            "webm": {"video/webm"},
            "mkv": {"video/x-matroska"},
        },
    }
    valid_types = allowed.get(media_kind, {})
    if extension not in valid_types:
        raise HTTPException(415, f"Недопустимый формат {media_kind}: разрешённые расширения {', '.join(valid_types)}")
    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if content_type and content_type != "application/octet-stream" and content_type not in valid_types[extension]:
        raise HTTPException(415, f"Тип файла не соответствует расширению .{extension}")

    total_bytes = 0
    content = bytearray()
    while chunk := await upload.read(1024 * 1024):
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            max_mb = max_bytes // (1024 * 1024)
            raise HTTPException(413, f"Размер файла превышает допустимые {max_mb} МБ")
        content.extend(chunk)
    return {"extension": extension, "size": total_bytes, "content": bytes(content)}


def convert_camera_video_to_mp4(
    video_bytes: bytes,
    max_bytes: int,
    source_extension: str = "webm",
) -> tuple[Path, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise HTTPException(503, "Сервис конвертации видео в MP4 недоступен.")
    if source_extension not in {"mp4", "webm"}:
        raise HTTPException(415, "Для записи с камеры поддерживаются только MP4 и WebM.")

    temp_dir = tempfile.mkdtemp(prefix="ukz-video-convert-")
    input_path = Path(temp_dir) / f"camera-recording-input.{source_extension}"
    output_path = Path(temp_dir) / "camera-recording-normalized.mp4"
    try:
        input_path.write_bytes(video_bytes)
        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
            raise HTTPException(422, "Не удалось преобразовать запись с камеры в MP4. Запишите видео ещё раз.")
        if output_path.stat().st_size > max_bytes:
            raise HTTPException(413, "Размер преобразованного MP4 превышает допустимые 100 МБ.")
        return output_path, temp_dir
    except subprocess.TimeoutExpired as error:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(422, "Конвертация видео заняла слишком много времени. Запишите более короткий ролик.") from error
    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(422, "Не удалось прочитать или преобразовать запись с камеры.") from error


def calculate_health(input_data: HealthCalculationInput) -> dict[str, Any]:
    m = input_data.measurements
    body = max(0.0, min(1.0, (1 - abs(m.resting_pulse - 70) / 80) * 0.55 + min(m.sleep_hours / 8, 1) * 0.45))
    mental = max(0.0, min(1.0, (1 - m.stress_score / 10) * 0.6 + m.verification_quality * 0.4))
    social = max(0.0, min(1.0, min(m.activity_minutes / 150, 1) * 0.65 + (1 - abs(m.respiratory_rate - 16) / 30) * 0.35))
    components = {"body": round(body, 3), "mental": round(mental, 3), "social": round(social, 3)}
    score = round(0.4 * body + 0.35 * mental + 0.25 * social, 3)
    completeness = round(sum(value is not None for value in m.model_dump().values()) / len(m.model_dump()), 2)
    if completeness < 0.7:
        category = "review"
    elif score >= 0.72:
        category = "stable"
    else:
        category = "attention"
    evidence = [
        {"key": "body", "label": "Физическое состояние", "value": f"{body:.2f}", "direction": "positive" if body >= 0.7 else "neutral"},
        {"key": "mental", "label": "Психоэмоциональная нагрузка", "value": f"{mental:.2f}", "direction": "positive" if mental >= 0.7 else "negative"},
        {"key": "social", "label": "Активность и восстановление", "value": f"{social:.2f}", "direction": "positive" if social >= 0.7 else "neutral"},
        {"key": "completeness", "label": "Полнота данных", "value": f"{completeness:.0%}", "direction": "positive" if completeness >= 0.9 else "neutral"},
    ]
    return {"health_id": score, "category": category, "status": "review_required" if category == "review" else "calculated", "components": components, "evidence": evidence}


def seed_demo(db: Session) -> None:
    if db.scalar(select(func.count()).select_from(HealthResult)):
        return
    now = datetime.utcnow()
    demo_inputs = [
        ("EMP-014", 0.86, "stable", "Результат в зелёной зоне, полнота данных 100%"),
        ("EMP-022", 0.68, "attention", "Повышенная психоэмоциональная нагрузка"),
        ("EMP-007", 0.57, "review", "Неполный набор измерений требует проверки"),
    ]
    for index, (employee_ref, score, category, description) in enumerate(demo_inputs):
        db.add(HealthResult(
            employee_ref=employee_ref,
            event_date=now - timedelta(hours=index * 4),
            health_id=score,
            category=category,
            status="review_required" if category == "review" else "calculated",
            model_version=get_settings().model_version,
            components={"body": round(score + 0.04, 2), "mental": round(score - 0.02, 2), "social": round(score, 2)},
            evidence=[{"key": "summary", "label": "Интерпретация", "value": description, "direction": "negative" if category != "stable" else "positive"}],
            disclaimer=DISCLAIMER,
            created_at=now - timedelta(hours=index * 4),
        ))
    db.add(VerificationSession(employee_ref="EMP-014", status="verified", match_score=0.94, liveness_score=0.89, quality_score=0.93, decision="match", algorithm_version="ukz-1.2.0", audit_ref="AUD-2026-0929-014", is_match=True, is_live=True, cosine_distance=0.94))
    db.add(VerificationSession(employee_ref="EMP-022", status="manual_review", match_score=0.72, liveness_score=0.66, quality_score=0.87, decision="review", algorithm_version="ukz-1.2.0", audit_ref="AUD-2026-0929-022", is_match=False, is_live=False, cosine_distance=0.72))
    db.flush()
    demo_verification = db.scalar(
        select(VerificationSession).where(VerificationSession.employee_ref == "EMP-022").order_by(VerificationSession.created_at.desc())
    )
    db.add(ReviewTask(kind="verification", subject_ref="EMP-022", status="open", priority="high", summary="Пограничное совпадение лица: 0.72. Требуется ручная проверка.", verification_session_id=demo_verification.id if demo_verification else None))
    db.add(ReviewTask(kind="health_id", subject_ref="EMP-007", status="open", priority="medium", summary="Полнота данных ниже 70%. Проверьте входное событие ПрМО."))
    for title, description, status in [
        ("Расчёт HEALTH_ID завершён", "Для EMP-014 получен результат 0.86", "success"),
        ("Новая задача в очереди", "EMP-022 направлен на ручную проверку", "attention"),
        ("Контроль качества данных", "Последняя партия событий прошла с полнотой 94%", "success"),
    ]:
        db.add(ActivityItem(type="system", title=title, description=description, status=status, created_at=now))
    db.commit()