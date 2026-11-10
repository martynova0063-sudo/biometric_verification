import asyncio
import cv2
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from dataclasses import fields
import math
from mimetypes import guess_type
from pathlib import Path
import shutil
import time
from typing import Any, Literal
from uuid import UUID, uuid4
import numpy as np

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from .authentication import (
    auth_required_response,
    load_bearer_user,
    load_session_user,
    router as auth_router,
)
from .biometrics import (
    InsufficientReferenceQualityError,
    analyze_biometric_video,
    build_insufficient_quality_result,
    build_not_evaluable_result,
)
from .config import get_settings
from .db import SessionLocal, get_db
from .drift_detection import (
    get_data_drift_summary,
    list_data_drift_runs,
    record_drift_observation,
    run_data_drift_analysis,
    update_manual_review_mode,
)
from .embedding_interpretation import (
    EmbeddingInterpretationGenerationError,
    EmbeddingInterpretationNotConfigured,
    generate_embedding_interpretation,
)
from .employee_import import (
    employee_has_photo,
    employee_photo_path,
    is_private_employee_photo,
    sync_employees,
)
from .employee_photo_archive import (
    EmployeePhotoArchiveError,
    archive_current_employee_photo,
)
from .health_id_eligibility import latest_health_id_eligible_verification
from .metrics import calculate_metrics_snapshot
from .personal_baseline import (
    calculate_personal_baseline,
    evaluate_result_list_baselines,
)
from .models import (
    ActivityItem,
    BiometricMetricsLog,
    DataDriftControl,
    Employee,
    EmployeePhotoUpload,
    HealthResult,
    RegisteredModel,
    ReviewTask,
    VerificationSession,
)
from .schemas import (
    ActivityItemResponse,
    BiometricMetricsResponse,
    DataDriftControlInput,
    DataDriftRunResponse,
    DataDriftSummaryResponse,
    DashboardSummary,
    EmbeddingInterpretationRequest,
    EmbeddingInterpretationResponse,
    EligibleHealthEmployeeResponse,
    EmployeePhotoFinalizeInput,
    EmployeePhotoUploadUrlInput,
    EmployeePhotoUploadUrlResponse,
    EmployeeResponse,
    HealthCalculationInput,
    HealthResultListResponse,
    HealthResultResponse,
    HealthStatus,
    PersonalBaselineResponse,
    ModelCard,
    ModelComponent,
    ModelVersion,
    ModelRegistryInput,
    ModelRegistryResponse,
    ReviewInput,
    ReviewTaskResponse,
    VerificationReviewDecisionInput,
    VerificationReviewQueueItemResponse,
    VerificationSessionStatusResponse,
    VerificationSessionDetailResponse,
    VerificationSessionResponse,
    VideoReceiptCompleteResponse,
)
from .object_storage import (
    MAX_PHOTO_BYTES,
    ObjectStorageError,
    fetch_private_employee_photo,
    sign_employee_photo_upload,
)
from .arcface_service import (
    ArcFaceImageError,
    ArcFaceIndexError,
    ArcFaceSearchService,
)
from .services import DISCLAIMER, calculate_health, convert_camera_video_to_mp4, inspect_upload, seed_demo
from .verification_config import (
    MODEL_CONFIG,
    PARAMETER_OPTION_VALUES,
    ModelConfig,
    parameter_schema,
)
settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version, docs_url="/api/docs", openapi_url="/api/openapi.json",
openapi_tags=[
                  {
                      "name": "verification",
                      "description": "Операции, связанные с биометрической верификацией",
                      "ru_name": "Верификация",
                      "ru_description": "Работа с сессиями биометрической верификации и результатами проверок"
                  },
    {
          "name": "drift",
          "description": "Детектирование дрейфа данных",
          "ru_name": "Дрейфа данных",
          "ru_description": "Работа с модулём дрейфа данных"
      },
    {
          "name": "employees",
          "description": "Сотрудники",
          "ru_name": "Сотрудники",
          "ru_description": "Работа с сотрудниками"
      },
    {
          "name": "governance",
          "description": "Управление",
          "ru_name": "Управление",
          "ru_description": "Работа с модулем управление"
      },
    {
          "name": "review",
          "description": "Ручное управление",
          "ru_name": "Ручное управление, задачи управления",
          "ru_description": "Работа с задачими и ручным подтверждением"
      },
    {
          "name": "health",
          "description": "Расчёт индекса Health_id",
          "ru_name": "Расчёт индекса Health_id",
          "ru_description": "Расчёт индекса здоровья"
                  },
    {
          "name": "auth",
          "description": "Browser authentication",
          "ru_name": "Вход",
          "ru_description": "Вход и завершение сессии"
      }
              ]
             )
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    session_cookie="oidc_state",
    max_age=10 * 60,
    same_site="lax",
    https_only=True,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
logger = logging.getLogger(__name__)
_drift_worker_task: asyncio.Task | None = None

arcface_search = ArcFaceSearchService(
    storage_dir=Path(__file__).resolve().parent / "storage"
)


@app.get("/api/arcface/status", tags=["verification"])
def arcface_status() -> dict[str, Any]:
    return arcface_search.status()


@app.post("/api/refresh-index", tags=["verification"])
def refresh_arcface_index() -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        index_stats = arcface_search.load_index()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ArcFaceIndexError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "status": "success",
        **index_stats,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "message": "Индекс сотрудников успешно обновлён.",
    }


@app.post(
    "/api/employees/{employee_guid}/arcface-index",
    tags=["employees", "verification"],
    summary="Добавить эталонное фото сотрудника в индекс ArcFace",
)
async def add_employee_to_arcface_index(
    employee_guid: UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    employee = db.get(Employee, str(employee_guid))
    if not employee:
        raise HTTPException(status_code=404, detail="Сотрудник не найден.")
    if not employee_has_photo(employee):
        raise HTTPException(
            status_code=404,
            detail="У сотрудника нет сохранённого эталонного фото.",
        )

    photo_bytes: bytes | None = None
    if is_private_employee_photo(employee.photo_path, employee.guid):
        try:
            photo_bytes = await fetch_private_employee_photo(employee.photo_path)
        except ObjectStorageError as error:
            logger.exception("Failed to load employee photo for ArcFace enrollment")
            raise HTTPException(
                status_code=503,
                detail="Не удалось получить эталонное фото сотрудника.",
            ) from error
    else:
        photo_path = employee_photo_path(employee.photo_path)
        if not photo_path or not photo_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Сохранённое эталонное фото сотрудника недоступно.",
            )
        try:
            if photo_path.stat().st_size > ArcFaceSearchService.max_image_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Размер эталонного фото не должен превышать 10 МБ.",
                )
            photo_bytes = await run_in_threadpool(photo_path.read_bytes)
        except OSError as error:
            raise HTTPException(
                status_code=503,
                detail="Не удалось прочитать эталонное фото сотрудника.",
            ) from error

    if not photo_bytes:
        raise HTTPException(
            status_code=404,
            detail="Сохранённое эталонное фото сотрудника недоступно.",
        )

    try:
        result = await run_in_threadpool(
            arcface_search.add_employee_image,
            photo_bytes,
            employee.guid,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ArcFaceImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ArcFaceIndexError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OSError as error:
        logger.exception("Failed to persist ArcFace enrollment")
        raise HTTPException(
            status_code=503,
            detail="Не удалось сохранить эмбеддинг в индекс ArcFace.",
        ) from error

    if result["already_indexed"]:
        message = "Это эталонное фото уже учтено в индексе."
    elif result["updated"]:
        message = "Эмбеддинг эталонного фото сотрудника обновлён."
    else:
        message = "Эмбеддинг эталонного фото сотрудника добавлен в индекс."
    return {
        **result,
        "employee_guid": employee.guid,
        "personnel_number": employee.personnel_number,
        "message": message,
    }


@app.post("/api/recognize", tags=["verification"])
async def recognize_employee(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Загрузите файл изображения.")

    try:
        contents = await file.read(ArcFaceSearchService.max_image_bytes + 1)
    finally:
        await file.close()
    if len(contents) > ArcFaceSearchService.max_image_bytes:
        raise HTTPException(status_code=413, detail="Размер изображения не должен превышать 10 МБ.")

    try:
        result = arcface_search.recognize_image(
            contents,
            top_k=top_k,
            threshold=threshold,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ArcFaceImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ArcFaceIndexError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    employee_uuids = [candidate["uuid"] for candidate in result["results"]]
    employees = db.scalars(
        select(Employee).where(Employee.guid.in_(employee_uuids))
    ).all() if employee_uuids else []
    employee_by_uuid = {employee.guid: employee for employee in employees}
    for candidate in result["results"]:
        employee = employee_by_uuid.get(candidate["uuid"])
        if employee:
            candidate["employee_ref"] = employee.personnel_number
            candidate["full_name"] = " ".join(
                part
                for part in (
                    employee.surname,
                    employee.given_name,
                    employee.patronymic,
                )
                if part
            )
    result["top_match"] = result["results"][0] if result["results"] else None
    result["disclaimer"] = (
        "Экспериментальный поиск кандидатов по сходству лица. "
        "Оценка не подтверждает личность и не должна использоваться "
        "как единственное основание для доступа."
    )
    return result


@app.middleware("http")
async def authenticate_api_requests(request: Request, call_next):
    path = request.url.path
    public_paths = {
        "/api/healthz",
        "/api/login",
        "/api/auth/login",
        "/api/callback",
        "/api/logout",
    }
    if request.method == "OPTIONS" or path in public_paths:
        return await call_next(request)
    if not path.startswith("/api/"):
        return await call_next(request)

    request.state.auth_user = None
    request.state.auth_user = load_bearer_user(
        request.headers.get("authorization")
    )
    if request.state.auth_user is None:
        db = SessionLocal()
        try:
            request.state.auth_user = load_session_user(
                db, request.cookies.get("sid")
            )
        finally:
            db.close()

    if path != "/api/auth/user" and request.state.auth_user is None:
        return auth_required_response()
    return await call_next(request)


def verification_session_payload(session: VerificationSession) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": session.id,
        "employee_ref": session.employee_ref,
        "status": session.status,
        "match_score": session.match_score,
        "liveness_score": session.liveness_score,
        "quality_score": session.quality_score,
        "decision": session.decision,
        "algorithm_version": session.algorithm_version,
        "created_at": session.created_at,
        "audit_ref": session.audit_ref,
        "attack_type": session.attack_type,
        "is_match": session.is_match,
        "ground_truth_match": session.ground_truth_match,
        "is_live": session.is_live,
        "ground_truth_liveness": session.ground_truth_liveness,
        "cosine_distance": session.cosine_distance,
        "embedding_comparison": session.embedding_comparison,
        "verification_elapsed_seconds": session.verification_elapsed_seconds,
    }
    payload.update({
        "verification_id": session.id,
        "event_id": "",
        "worker_pseudonym": session.employee_ref,
        "model_version": session.algorithm_version,
        "config_hash": "",
        "face_match": {},
        "liveness": {},
        "quality": {},
        "route": session.status,
        "health_id": {},
        "review_tasks": [],
        "disclaimer": DISCLAIMER,
    })
    payload.update(session.details or {})
    return payload


def sync_review_task_details(
    session: VerificationSession,
    task: ReviewTask,
    *,
    trigger: str,
) -> None:
    details = dict(session.details or {})
    review_tasks = list(details.get("review_tasks") or [])
    existing_index = next(
        (
            index
            for index, item in enumerate(review_tasks)
            if isinstance(item, dict) and item.get("review_id") == task.id
        ),
        None,
    )
    if existing_index is None:
        review_tasks.append({
            "review_id": task.id,
            "calculation_id": session.id,
            "status": task.status,
            "priority": task.priority,
            "trigger": trigger,
            "trigger_details": {
                "system_decision": session.decision,
                "system_status": session.status,
                "match_score": session.match_score,
            },
            "match_score": session.match_score,
            "verification_route": session.status,
        })
    else:
        task_details = dict(review_tasks[existing_index])
        task_details.update({"status": task.status, "priority": task.priority})
        review_tasks[existing_index] = task_details
    details["review_tasks"] = review_tasks
    session.details = details


@app.on_event("startup")
def startup() -> None:
    global _drift_worker_task
    with next(get_db()) as db:
        seed_demo(db)
        sync_employees(db)
        if not db.scalar(select(RegisteredModel).where(RegisteredModel.status == "current")):
            db.add(RegisteredModel(
                model_name="HEALTH_ID",
                version=settings.model_version,
                status="current",
                purpose="Версионированный расчёт исследовательского индекса и биометрической верификации.",
                change_summary="Системная конфигурация по умолчанию из verification_config.py.",
                parameters=MODEL_CONFIG.public_settings(),
            ))
            db.commit()
    try:
        arcface_search.load_index()
    except (FileNotFoundError, ArcFaceIndexError):
        logger.warning("ArcFace employee index is unavailable.", exc_info=True)
    try:
        arcface_search.load_model()
    except (FileNotFoundError, ArcFaceIndexError):
        logger.warning("ArcFace model is unavailable.", exc_info=True)
    _drift_worker_task = asyncio.create_task(_daily_drift_worker())


async def _daily_drift_worker() -> None:
    while True:
        try:
            await run_in_threadpool(_run_scheduled_drift_analysis)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Daily data-drift analysis failed.")
        await asyncio.sleep(24 * 60 * 60)


def _run_scheduled_drift_analysis() -> None:
    with SessionLocal() as db:
        run_data_drift_analysis(db)


@app.on_event("shutdown")
async def shutdown() -> None:
    if _drift_worker_task:
        _drift_worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await _drift_worker_task


def validated_model_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    defaults = MODEL_CONFIG.public_settings()
    if set(parameters) != set(defaults):
        missing = sorted(set(defaults) - set(parameters))
        extra = sorted(set(parameters) - set(defaults))
        raise HTTPException(422, detail={"message": "Набор параметров не совпадает с системной конфигурацией.", "missing": missing, "extra": extra})

    normalized: dict[str, Any] = {}
    for field in fields(ModelConfig):
        key = field.name
        value = parameters[key]
        if key == "multi_face_strategy" and value in {"skip", "sckip"}:
            value = "skip_frame"
        default = defaults[key]
        if isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = type(value) is int
        elif isinstance(default, float):
            valid = type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value)
        elif isinstance(default, str):
            valid = isinstance(value, str) and bool(value.strip())
        elif isinstance(default, tuple):
            valid = isinstance(value, (list, tuple)) and len(value) == len(default)
            if valid:
                valid = all(
                    type(item) is type(default_item) or
                    (isinstance(default_item, float) and type(item) is int)
                    for item, default_item in zip(value, default)
                )
                valid = valid and all(
                    math.isfinite(float(item)) if isinstance(item, (int, float)) else True
                    for item in value
                )
                value = tuple(float(item) if isinstance(default_item, float) else int(item) for item, default_item in zip(value, default))
        else:
            valid = False
        if not valid:
            raise HTTPException(422, detail=f"Некорректное значение параметра «{key}».")
        if key in PARAMETER_OPTION_VALUES and value not in PARAMETER_OPTION_VALUES[key]:
            raise HTTPException(422, detail=f"Недопустимый вариант параметра «{key}».")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise HTTPException(422, detail=f"Параметр «{key}» не может быть отрицательным.")
        if isinstance(value, tuple) and any(item < 0 for item in value):
            raise HTTPException(422, detail=f"Параметр «{key}» не может содержать отрицательные значения.")
        if key == "min_resolution" and any(item < 1 for item in value):
            raise HTTPException(422, detail="Минимальное разрешение должно быть не меньше 1×1.")
        if key in {"max_frames_to_process", "frame_sample_interval", "liveness_frame_sample_interval", "frame_median_count", "min_detected_frames", "max_faces_per_frame", "blink_min_count", "head_motion_min_frames"} and value < 1:
            raise HTTPException(422, detail=f"Параметр «{key}» должен быть не меньше 1.")
        if (key.endswith("_threshold") or key.endswith("_weight") or key in {"match_threshold", "manual_review_threshold", "liveness_pass", "liveness_fail", "blink_min_ratio", "liveness_rppg_min_confidence"}) and isinstance(value, (int, float)) and not 0 <= value <= 1:
            raise HTTPException(422, detail=f"Параметр «{key}» должен быть в диапазоне от 0 до 1.")
        normalized[key] = value

    if normalized["match_threshold"] <= normalized["manual_review_threshold"]:
        raise HTTPException(422, detail="Порог совпадения должен быть выше порога ручной проверки.")
    if normalized["liveness_pass"] <= normalized["liveness_fail"]:
        raise HTTPException(422, detail="Порог подтверждения живого присутствия должен быть выше порога отказа.")
    if normalized["multi_face_warn_threshold"] > normalized["multi_face_block_threshold"]:
        raise HTTPException(
            422,
            detail="Порог предупреждения о нескольких лицах не может превышать порог блокировки.",
        )
    if normalized["frame_median_count"] not in {5, 10}:
        raise HTTPException(422, detail="Число кадров для медианы должно быть 5 или 10.")
    if normalized["frame_quality_min_brightness"] >= normalized["frame_quality_max_brightness"]:
        raise HTTPException(422, detail="Минимальная яркость должна быть ниже максимальной.")
    if normalized["frame_quality_max_illumination_spread"] > 255:
        raise HTTPException(422, detail="Неравномерность освещения должна быть не выше 255.")
    if normalized["liveness_strategy"] not in PARAMETER_OPTION_VALUES["liveness_strategy"]:
        raise HTTPException(
            422,
            detail="Стратегия живого присутствия должна быть weighted_avg, and, or или adaptive.",
        )
    if normalized["liveness_method"] not in PARAMETER_OPTION_VALUES["liveness_method"]:
        raise HTTPException(
            422,
            detail="Метод живого присутствия должен быть siamese, rppg или ensemble.",
        )
    if sum(
        normalized[key]
        for key in (
            "blink_weight",
            "head_motion_weight",
            "texture_weight",
            "depth_weight",
            "liveness_rppg_weight",
            "liveness_siamese_weight",
        )
    ) <= 0:
        raise HTTPException(422, detail="Хотя бы один вес компонентов живого присутствия должен быть больше нуля.")
    return ModelConfig(**normalized).public_settings()


def model_payload(model: RegisteredModel) -> ModelRegistryResponse:
    return ModelRegistryResponse(
        id=model.id,
        model_name=model.model_name,
        version=model.version,
        status=model.status,
        purpose=model.purpose,
        change_summary=model.change_summary,
        parameters={**MODEL_CONFIG.public_settings(), **(model.parameters or {})},
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def active_model_or_503(db: Session) -> RegisteredModel:
    model = db.scalar(select(RegisteredModel).where(RegisteredModel.status == "current"))
    if model is None:
        raise HTTPException(503, "Активная модель не настроена.")
    return model


@app.get("/api/healthz", summary="Проверка рабоспособности сервиса", response_model=HealthStatus)
def health_check() -> HealthStatus:
    return HealthStatus(status="ok", service=settings.app_name, version=settings.app_version)


@app.get("/api/dashboard/summary", summary="Получение сводной статистики/данных для дашборда", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    total = db.scalar(select(func.count()).select_from(HealthResult)) or 0
    today = db.scalar(select(func.count()).select_from(HealthResult).where(func.date(HealthResult.created_at) == datetime.utcnow().date())) or 0
    verification_total = db.scalar(
        select(func.count()).select_from(VerificationSession).where(
            VerificationSession.status.not_in(["not_evaluable", "insufficient_quality"])
        )
    ) or 0
    verified = db.scalar(select(func.count()).select_from(VerificationSession).where(VerificationSession.status == "verified")) or 0
    open_tasks = db.scalar(select(func.count()).select_from(ReviewTask).where(ReviewTask.status != "resolved")) or 0
    categories = {category: db.scalar(select(func.count()).select_from(HealthResult).where(HealthResult.category == category)) or 0 for category in ["stable", "attention", "review"]}
    return DashboardSummary(total_events=total, health_id_today=today, verification_pass_rate=round(verified / verification_total, 2) if verification_total else 0, open_review_tasks=open_tasks, data_quality=0.94, model_version=settings.model_version, categories=categories)


@app.get("/api/activity", summary="Получение списка активностей (журнала действий)", response_model=list[ActivityItemResponse])
def list_activity(limit: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)) -> list[ActivityItem]:
    return list(db.scalars(select(ActivityItem).order_by(ActivityItem.created_at.desc()).limit(limit)))


@app.get("/api/health-results", tags=["health"], summary="Получение списка результатов расчёта Health ID", response_model=list[HealthResultListResponse])
def list_health_results(category: str = Query("all"), limit: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    stmt = select(HealthResult).order_by(HealthResult.created_at.desc()).limit(limit)
    if category != "all":
        stmt = stmt.where(HealthResult.category == category)
    results = list(db.scalars(stmt))
    baseline_by_result = evaluate_result_list_baselines(db, results)
    return [
        {
            **HealthResultResponse.model_validate(result).model_dump(),
            "personal_baseline": baseline_by_result[result.id],
        }
        for result in results
    ]


@app.get(
    "/api/health-results/eligible-employees",
    tags=["health"],
    summary="Список сотрудников с успешной или вручную подтверждённой верификацией",
    response_model=list[EligibleHealthEmployeeResponse],
)
def list_eligible_health_employees(
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    employees = db.scalars(
        select(Employee).order_by(
            Employee.surname,
            Employee.given_name,
            Employee.patronymic,
        )
    )
    eligible = []
    for employee in employees:
        result = latest_health_id_eligible_verification(db, employee.personnel_number)
        if result is None:
            continue
        session, source, review_task = result
        eligible.append({
            "employee_ref": employee.personnel_number,
            "full_name": " ".join(
                part
                for part in (
                    employee.surname,
                    employee.given_name,
                    employee.patronymic,
                )
                if part
            ),
            "organization": employee.organization,
            "is_test": employee.is_test,
            "verification_session_id": session.id,
            "verification_status": session.status,
            "eligibility_source": source,
            "verified_at": (
                review_task.resolved_at or session.created_at
                if source == "manual" and review_task
                else session.created_at
            ),
        })
    return eligible


@app.get(
    "/api/health-results/employees/{employee_ref}/personal-baseline",
    tags=["health"],
    summary="Рассчитать персональные коридоры показателей за последние 30 дней",
    response_model=PersonalBaselineResponse,
)
def get_personal_health_baseline(
    employee_ref: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    employee = db.scalar(
        select(Employee).where(Employee.personnel_number == employee_ref)
    )
    if employee is None:
        raise HTTPException(404, "Сотрудник не найден.")
    return calculate_personal_baseline(db, employee_ref)


@app.post("/api/health-results", tags=["health"], summary="Запуск расчёта Health ID", response_model=HealthResultResponse, status_code=201)
def calculate_health_id(payload: HealthCalculationInput, db: Session = Depends(get_db)) -> HealthResult:
    employee = db.scalar(select(Employee).where(Employee.personnel_number == payload.employee_ref))
    if employee is None:
        raise HTTPException(404, "Сотрудник не найден.")
    if latest_health_id_eligible_verification(db, payload.employee_ref) is None:
        raise HTTPException(
            409,
            "Для расчёта HEALTH_ID требуется последняя успешная или вручную подтверждённая верификация.",
        )

    result_data = calculate_health(payload)
    result_id = str(uuid4())
    result = HealthResult(
        id=result_id,
        employee_ref=payload.employee_ref,
        event_date=payload.event_date,
        model_version=settings.model_version,
        disclaimer=DISCLAIMER,
        measurements=payload.measurements.model_dump(
            exclude={"verification_quality"}
        ),
        **result_data,
    )
    measurements = payload.measurements
    record_drift_observation(
        db,
        source_type="health_id",
        observed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        is_test=bool(employee.is_test) if employee else True,
        metrics={
            "heart_rate": measurements.resting_pulse,
            "breathing_rate": measurements.respiratory_rate,
            "stress_index": measurements.stress_score,
            "health_id_value": result_data.get("health_id"),
        },
    )
    db.add(result)
    db.add(ActivityItem(type="health_id", title="Расчёт HEALTH_ID завершён", description=f"Для {payload.employee_ref} получен результат {result_data['health_id']:.2f}", status="success"))
    if result_data["status"] == "review_required":
        db.add(ReviewTask(kind="health_id", subject_ref=payload.employee_ref, priority="medium", summary="Результат требует проверки из-за неполноты данных."))
    db.commit()
    db.refresh(result)
    return result


@app.get("/api/health-results/{result_id}", tags=["health"], summary="Получение детальных данных по конкретному результату расчёта Health ID (по идентификатору)", response_model=HealthResultResponse)
def get_health_result(result_id: UUID, db: Session = Depends(get_db)) -> HealthResult:
    result = db.get(HealthResult, str(result_id))
    if not result:
        raise HTTPException(404, "Результат не найден")
    return result


@app.get("/api/verification/sessions", summary="Получение списка сессий биометрической верификации", 
         tags=["verification"],
         response_model=list[VerificationSessionResponse])
def list_verification_sessions(limit: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)) -> list[VerificationSession]:
    return list(db.scalars(select(VerificationSession).order_by(VerificationSession.created_at.desc()).limit(limit)))


@app.post(
    "/api/verification/sessions/{session_id}/embedding-interpretation",
    summary="Сформировать техническую интерпретацию агрегатов эмбеддингов",
    tags=["verification"],
    response_model=EmbeddingInterpretationResponse,
)
async def create_embedding_interpretation(
    session_id: UUID,
    payload: EmbeddingInterpretationRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = db.get(VerificationSession, str(session_id))
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    details = dict(session.details) if isinstance(session.details, dict) else {}
    comparison = details.get("embedding_comparison")
    if not isinstance(comparison, dict) or comparison.get("status") != "available":
        raise HTTPException(
            409,
            "Для этой сессии нет достаточных агрегированных данных для интерпретации.",
        )

    cached = details.get("embedding_interpretation")
    if (
        not payload.refresh
        and isinstance(cached, dict)
        and cached.get("analysis_version") == 1
    ):
        return cached

    try:
        interpretation = await generate_embedding_interpretation(comparison)
    except EmbeddingInterpretationNotConfigured as error:
        raise HTTPException(
            503,
            "Сервис интерпретации LLM не настроен.",
        ) from error
    except EmbeddingInterpretationGenerationError as error:
        raise HTTPException(
            502,
            "Не удалось получить интерпретацию LLM. Попробуйте позже.",
        ) from error

    details["embedding_interpretation"] = interpretation
    session.details = details
    db.add(session)
    db.commit()
    return interpretation


@app.post(
    "/api/verification/videos/convert",
    summary="Преобразовать запись с камеры в MP4",
    tags=["verification"],
    responses={
        200: {
            "content": {
                "video/mp4": {
                    "schema": {"type": "string", "format": "binary"},
                },
            },
        },
    },
)
async def convert_verification_camera_video(
    video_file: UploadFile = File(..., description="Запись с камеры в формате WebM или MP4"),
) -> FileResponse:
    try:
        video_meta = await inspect_upload(video_file, "video", max_bytes=100 * 1024 * 1024)
    finally:
        await video_file.close()

    if video_meta["extension"] not in {"webm", "mp4"}:
        raise HTTPException(415, "Для записи с камеры поддерживаются только MP4 и WebM.")

    output_path, temp_dir = await run_in_threadpool(
        convert_camera_video_to_mp4,
        video_meta["content"],
        100 * 1024 * 1024,
        video_meta["extension"],
    )
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="camera-recording.mp4",
        background=BackgroundTask(shutil.rmtree, temp_dir, ignore_errors=True),
    )


@app.get("/api/employees", tags=["employees"], response_model=list[EmployeeResponse], summary="Список сотрудников с эталонными фото")
def list_employees(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    employees = db.scalars(
        select(Employee).order_by(
            Employee.is_test.desc(),
            Employee.organization,
            Employee.surname,
            Employee.given_name,
            Employee.patronymic,
        )
    )
    payload = []
    for employee in employees:
        has_photo = employee_has_photo(employee)
        payload.append({
            "guid": employee.guid,
            "personnel_number": employee.personnel_number,
            "surname": employee.surname,
            "given_name": employee.given_name,
            "patronymic": employee.patronymic,
            "organization": employee.organization,
            "is_test": employee.is_test,
            "has_photo": has_photo,
            "photo_url": f"/api/employees/{employee.guid}/photo" if has_photo else None,
        })
    return payload


@app.post(
    "/api/employees/{employee_guid}/photo/upload-url",
    tags=["employees"],
    response_model=EmployeePhotoUploadUrlResponse,
    summary="Создать защищённую ссылку для загрузки снимка сотрудника",
)
async def request_employee_photo_upload(
    employee_guid: UUID,
    payload: EmployeePhotoUploadUrlInput,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    guid = str(employee_guid)
    employee = db.get(Employee, guid)
    if employee is None:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")
    try:
        upload_url, object_path = await sign_employee_photo_upload(guid)
    except ObjectStorageError as error:
        logger.exception("Failed to create employee photo upload URL")
        raise HTTPException(status_code=503, detail="Защищённое хранилище временно недоступно") from error

    now = datetime.utcnow()
    db.execute(
        delete(EmployeePhotoUpload).where(
            EmployeePhotoUpload.expires_at < now
        )
    )
    db.add(
        EmployeePhotoUpload(
            employee_guid=guid,
            user_id=request.state.auth_user["id"],
            object_path=object_path,
            content_type=payload.content_type,
            expected_size=payload.size,
            expires_at=now + timedelta(minutes=15),
        )
    )
    db.commit()
    return {"upload_url": upload_url, "object_path": object_path}


@app.post(
    "/api/employees/{employee_guid}/photo",
    tags=["employees"],
    response_model=EmployeeResponse,
    summary="Сохранить загруженный снимок как эталонное фото сотрудника",
)
async def save_employee_photo(
    employee_guid: UUID,
    payload: EmployeePhotoFinalizeInput,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    guid = str(employee_guid)
    employee = db.scalar(
        select(Employee).where(Employee.guid == guid).with_for_update()
    )
    if employee is None:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")
    replacing_photo = employee_has_photo(employee)

    upload = db.scalar(
        select(EmployeePhotoUpload).where(
            EmployeePhotoUpload.employee_guid == guid,
            EmployeePhotoUpload.user_id == request.state.auth_user["id"],
            EmployeePhotoUpload.object_path == payload.object_path,
        )
    )
    if upload is None:
        raise HTTPException(status_code=404, detail="Загрузка фото не найдена или недоступна")
    if upload.expires_at <= datetime.utcnow():
        db.delete(upload)
        db.commit()
        raise HTTPException(status_code=410, detail="Срок загрузки фото истёк")

    try:
        photo_bytes = await fetch_private_employee_photo(upload.object_path)
    except ObjectStorageError as error:
        logger.exception("Failed to read uploaded employee photo")
        raise HTTPException(status_code=503, detail="Не удалось проверить снимок в хранилище") from error
    if not photo_bytes:
        raise HTTPException(status_code=422, detail="Снимок ещё не загружен")
    if len(photo_bytes) != upload.expected_size or len(photo_bytes) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=422, detail="Размер загруженного снимка не совпадает")
    if not photo_bytes.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=415, detail="Загруженный файл не является JPEG-снимком")

    decoded = cv2.imdecode(np.frombuffer(photo_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=415, detail="Не удалось прочитать JPEG-снимок")
    height, width = decoded.shape[:2]
    if width < 120 or height < 120 or width > 8192 or height > 8192:
        raise HTTPException(status_code=422, detail="Разрешение снимка должно быть от 120 до 8192 пикселей")

    if replacing_photo:
        try:
            archived = await archive_current_employee_photo(
                employee,
                request.state.auth_user["id"],
                db,
            )
            if archived is None:
                raise EmployeePhotoArchiveError("The previous photo could not be archived.")
        except EmployeePhotoArchiveError as error:
            logger.exception("Could not preserve the previous employee photo")
            raise HTTPException(
                status_code=503,
                detail="Не удалось сохранить прежнее фото в dataset. Текущее фото не изменено.",
            ) from error

    employee.photo_path = upload.object_path
    employee.is_photo = True
    db.delete(upload)
    try:
        db.commit()
    except Exception as error:
        db.rollback()
        logger.exception("Could not commit employee photo replacement")
        raise HTTPException(
            status_code=503,
            detail="Не удалось сохранить новое фото. Текущее фото не изменено.",
        ) from error
    return {
        "guid": employee.guid,
        "personnel_number": employee.personnel_number,
        "surname": employee.surname,
        "given_name": employee.given_name,
        "patronymic": employee.patronymic,
        "organization": employee.organization,
        "is_test": employee.is_test,
        "has_photo": True,
        "photo_url": f"/api/employees/{employee.guid}/photo",
    }


@app.get("/api/employees/{employee_guid}/photo", tags=["employees"], summary="Эталонное фото сотрудника")
async def get_employee_photo(employee_guid: UUID, db: Session = Depends(get_db)) -> Response:
    employee = db.get(Employee, str(employee_guid))
    if not employee:
        raise HTTPException(404, "Сотрудник не найден")
    if is_private_employee_photo(employee.photo_path, employee.guid):
        try:
            photo_bytes = await fetch_private_employee_photo(employee.photo_path)
        except ObjectStorageError as error:
            logger.exception("Failed to serve private employee photo")
            raise HTTPException(status_code=503, detail="Эталонное фото временно недоступно") from error
        if not photo_bytes:
            raise HTTPException(404, "Эталонное фото сотрудника не загружено")
        return Response(
            content=photo_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
    photo_path = employee_photo_path(employee.photo_path)
    if not photo_path or not photo_path.is_file():
        raise HTTPException(404, "Эталонное фото сотрудника не загружено")
    return FileResponse(
        photo_path,
        media_type=guess_type(photo_path.name)[0] or "application/octet-stream",
        filename=photo_path.name,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/api/verification/sessions", tags=["verification"], summary="Создание новой сессии биометрической верификации", response_model=VerificationSessionDetailResponse, status_code=201)
async def create_verification_session(
    employee_ref: str = Form(..., min_length=1),
    photo_file: UploadFile = File(..., description="Фото для верификации"),
    video_file: UploadFile = File(..., description="Видео (обязательно)"),
    event_id: str = Form("", max_length=120),
    video_source: Literal["camera", "upload"] = Form("upload"),
    camera_model: str | None = Form(None, max_length=120),
    camera_width: int | None = Form(None, ge=1, le=8192),
    camera_height: int | None = Form(None, ge=1, le=8192),
    camera_fps: float | None = Form(None, ge=0, le=240),
    camera_facing_mode: Literal["user", "environment"] | None = Form(None),
    enroll_video_embeddings: bool = Form(False),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    verification_started_at = time.perf_counter()
    try:
        photo_meta = await inspect_upload(photo_file, "photo", max_bytes=10 * 1024 * 1024)
        video_meta = await inspect_upload(video_file, "video", max_bytes=100 * 1024 * 1024)
    finally:
        await photo_file.close()
        await video_file.close()
    active_model = active_model_or_503(db)
    active_parameters = {
        **MODEL_CONFIG.public_settings(),
        **(active_model.parameters or {}),
    }
    active_config = ModelConfig(**validated_model_parameters(active_parameters))
    verification_created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        result_data = await run_in_threadpool(
            analyze_biometric_video,
            photo_meta["content"],
            video_meta["content"],
            video_meta["extension"],
            active_config,
            active_model.version,
            arcface_embedding_callback=(
                arcface_search.embedding_from_face_crop
                if arcface_search.model is not None
                else None
            ),
            model_embedding_callback=(
                arcface_search.embedding_from_face_crop
                if arcface_search.model is not None
                else None
            ),
        )
    except InsufficientReferenceQualityError as error:
        result_data = build_insufficient_quality_result(
            str(error),
            active_config,
            active_model.version,
        )
    except HTTPException as error:
        if error.status_code not in {422, 503}:
            raise
        result_data = build_not_evaluable_result(
            str(error.detail),
            active_config,
            active_model.version,
        )
    except RuntimeError as error:
        if str(error) != "Required OpenCV Haar cascade files are unavailable":
            raise
        result_data = build_not_evaluable_result(
            "Компонент детекции лица недоступен.",
            active_config,
            active_model.version,
        )

    verification_id = str(uuid4())
    arcface_embedding_samples = result_data.pop("arcface_embedding_samples", [])
    result_data.pop("arcface_embedding_errors", None)
    result_details = result_data["details"]
    result_details.update({
        "verification_id": verification_id,
        "event_id": event_id,
        "worker_pseudonym": employee_ref,
        "review_tasks": [],
    })
    embedding_employee = db.scalar(
        select(Employee).where(Employee.personnel_number == employee_ref)
    )
    if embedding_employee:
        try:
            result_details["embedding_comparison"] = (
                arcface_search.compare_video_embeddings(
                    embedding_employee.guid,
                    arcface_embedding_samples,
                )
            )
        except Exception:
            logger.exception("ArcFace embedding comparison failed for verification session.")
            result_details["embedding_comparison"] = {
                "status": "unavailable",
                "method": "ArcFace ResNet50",
                "reference_vectors": 0,
                "video_vectors": len(arcface_embedding_samples),
                "dimensions": 512,
                "mean": {
                    "reference": None,
                    "video": None,
                    "delta": None,
                    "centroid_cosine_similarity": None,
                },
                "variance": {"reference": None, "video": None, "ratio": None},
                "outliers": {"count": None, "total": 0, "threshold": None},
                "distribution": {
                    "bin_centers": [],
                    "reference_counts": [],
                    "video_counts": [],
                },
                "message": "Не удалось рассчитать сравнение эмбеддингов.",
            }
    drift_control = db.get(DataDriftControl, 1)
    if enroll_video_embeddings:
        if (
            result_data.get("status") != "verified"
            or (drift_control and drift_control.manual_review_mode)
        ):
            result_details["arcface_video_enrollment"] = {
                "status": "skipped_verification_not_passed",
                "added_vectors": 0,
                "message": (
                    "Индекс не изменён: видео не прошло успешную верификацию."
                ),
            }
        elif embedding_employee is None:
            result_details["arcface_video_enrollment"] = {
                "status": "employee_not_found",
                "added_vectors": 0,
                "message": "Сотрудник не найден; эмбеддинги не добавлены.",
            }
        elif not arcface_embedding_samples:
            result_details["arcface_video_enrollment"] = {
                "status": "no_video_vectors",
                "added_vectors": 0,
                "message": (
                    "Не удалось получить эмбеддинги видео; индекс не изменён."
                ),
            }
        else:
            try:
                enrollment_result = await run_in_threadpool(
                    arcface_search.add_video_embeddings,
                    embedding_employee.guid,
                    arcface_embedding_samples,
                )
                result_details["arcface_video_enrollment"] = {
                    "status": (
                        "updated"
                        if enrollment_result["added_vectors"]
                        else "already_indexed"
                    ),
                    **enrollment_result,
                    "message": (
                        f"В индекс добавлено векторов: "
                        f"{enrollment_result['added_vectors']}."
                        if enrollment_result["added_vectors"]
                        else "Новых векторов нет: полученные признаки уже учтены."
                    ),
                }
            except Exception:
                logger.exception(
                    "ArcFace video enrollment failed for verification session."
                )
                result_details["arcface_video_enrollment"] = {
                    "status": "failed",
                    "added_vectors": 0,
                    "message": (
                        "Не удалось пополнить индекс. Верификация сохранена; "
                        "проверьте доступность модели и индекса."
                    ),
                }
    if drift_control and drift_control.manual_review_mode and result_data["status"] not in {"insufficient_quality", "not_evaluable"}:
        result_data["status"] = "manual_review"
        result_data["decision"] = "review"
        result_details["route"] = "manual_review"
        result_details["recommended_action"] = "manual_review_due_to_data_drift"
        result_details["drift_control"] = {
            "manual_review_mode": True,
            "reason": drift_control.manual_review_reason,
        }
    if result_data["status"] in {"manual_review", "insufficient_quality", "not_verified", "not_evaluable"}:
        priority = "high" if result_data["status"] in {"manual_review", "not_evaluable"} else "medium"
        if result_data["status"] == "not_evaluable":
            reason = result_details.get("not_evaluable_reason") or "причина не указана"
            summary = (
                f"Оценка невозможна: {reason}. Запросите повторную запись "
                "или выполните ручную проверку."
            )
        elif result_data["status"] == "insufficient_quality":
            quality_issues = result_details.get("quality", {}).get("blocking_issues", [])
            reason = "; ".join(quality_issues) or "качество входных данных недостаточно"
            summary = (
                f"Недостаточное качество входных данных: {reason}. Запросите повторную запись "
                "или выполните ручную проверку."
            )
        else:
            summary = f"Сессия верификации требует проверки: {result_data['status']}"
        review_task = ReviewTask(
            kind="verification",
            subject_ref=employee_ref,
            priority=priority,
            summary=summary[:500],
            verification_session_id=verification_id,
        )
        db.add(review_task)
        db.flush()
        result_details["review_tasks"] = [{
            "review_id": review_task.id,
            "calculation_id": verification_id,
            "status": review_task.status,
            "priority": review_task.priority,
            "trigger": result_data["status"],
            "trigger_details": {
                "match_score": result_data["match_score"],
                "liveness_score": result_data["liveness_score"],
                "quality_score": result_data["quality_score"],
                "not_evaluable_reason": result_details.get("not_evaluable_reason"),
                "recommended_action": result_details.get("recommended_action"),
            },
            "match_score": result_data["match_score"],
            "liveness_score": result_data["liveness_score"],
            "verification_route": result_data["status"],
        }]

    result_details["verification_elapsed_seconds"] = round(
        time.perf_counter() - verification_started_at,
        3,
    )
    session = VerificationSession(
        id=verification_id,
        employee_ref=employee_ref,
        status=result_data["status"],
        match_score=result_data["match_score"],
        liveness_score=result_data["liveness_score"],
        quality_score=result_data["quality_score"],
        decision=result_data["decision"],
        algorithm_version=result_details["model_version"],
        created_at=verification_created_at,
        audit_ref=f"AUD-{datetime.utcnow():%Y%m%d}-{uuid4().hex[:8].upper()}",
        attack_type=result_data["attack_type"],
        details=result_details,
        is_match=(
            None
            if result_data["status"] == "not_evaluable"
            or result_details["face_match"]["match_score"] is None
            else result_details["face_match"]["match_score"] >= result_details["face_match"]["threshold"]
        ),
        is_live=(
            None
            if result_data["status"] == "not_evaluable"
            or result_details["liveness"]["liveness_score"] is None
            else result_details["liveness"]["decision"] == "live"
        ),
        cosine_distance=(
            None
            if result_data["status"] == "not_evaluable"
            or result_details["face_match"]["match_score"] is None
            else result_details["face_match"]["match_score"]
        ),
    )
    employee = db.scalar(select(Employee).where(Employee.personnel_number == employee_ref))
    rppg = (
        (result_details.get("liveness") or {})
        .get("rppg", {})
        .get("details", {})
    )
    health_id_details = result_details.get("health_id")
    if isinstance(health_id_details, dict):
        health_id_value = health_id_details.get("health_id_value", health_id_details.get("value"))
    else:
        health_id_value = health_id_details
    quality = result_details.get("quality", {})
    quality_checks = quality.get("checks", {}) if isinstance(quality, dict) else {}
    lighting_check = quality_checks.get("lighting", {}) if isinstance(quality_checks, dict) else {}
    sharpness_check = quality_checks.get("blur", {}) if isinstance(quality_checks, dict) else {}
    record_drift_observation(
        db,
        source_type="verification",
        observed_at=verification_created_at,
        is_test=bool(employee.is_test) if employee else True,
        metrics={
            "match_score": result_data.get("match_score"),
            "liveness_score": result_data.get("liveness_score"),
            "quality_score": result_data.get("quality_score"),
            "heart_rate": rppg.get("heart_rate") if isinstance(rppg, dict) else None,
            "hrv": rppg.get("hrv") if isinstance(rppg, dict) else None,
            "breathing_rate": rppg.get("breathing_rate") if isinstance(rppg, dict) else None,
            "health_id_value": health_id_value,
            "lighting_brightness": lighting_check.get("actual") if isinstance(lighting_check, dict) else None,
            "image_sharpness": sharpness_check.get("actual") if isinstance(sharpness_check, dict) else None,
        },
        capture_metadata={
            "video_source": video_source,
            "camera_model": camera_model,
            "camera_width": camera_width,
            "camera_height": camera_height,
            "camera_fps": camera_fps,
            "camera_facing_mode": camera_facing_mode,
        },
    )
    db.add(session)
    db.add(ActivityItem(type="verification", title="Сессия верификации обработана", description=f"{employee_ref}: {result_data['status']}", status="success" if result_data["status"] == "verified" else "attention"))
    db.commit()
    db.refresh(session)
    return verification_session_payload(session)


@app.get(
    "/api/drift/summary",
    summary="Сводка мониторинга дрейфа данных",
    response_model=DataDriftSummaryResponse,
    tags=["drift"],
)
def get_drift_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    return get_data_drift_summary(db)


@app.get(
    "/api/drift/history",
    summary="История ежедневных проверок дрейфа",
    response_model=list[DataDriftRunResponse],
    tags=["drift"],
)
def get_drift_history(
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[Any]:
    return list_data_drift_runs(db, limit)


@app.post(
    "/api/drift/run",
    summary="Запустить анализ дрейфа немедленно",
    response_model=DataDriftSummaryResponse,
    tags=["drift"],
)
def run_drift_analysis(db: Session = Depends(get_db)) -> dict[str, Any]:
    return run_data_drift_analysis(db)


@app.post(
    "/api/drift/control",
    summary="Изменить режим ручной проверки",
    response_model=DataDriftSummaryResponse,
    tags=["drift"],
)
def set_drift_control(
    payload: DataDriftControlInput,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        summary = update_manual_review_mode(
            db,
            enabled=payload.enabled,
            reviewer_note=payload.reviewer_note,
        )
    except ValueError as error:
        raise HTTPException(409, detail=str(error)) from error
    db.add(ActivityItem(
        type="model",
        title="Изменён режим обработки при дрейфе",
        description=(
            f"Ручная проверка {'включена' if payload.enabled else 'отключена'}: "
            f"{payload.reviewer_note.strip()}"
        )[:500],
        status="attention",
    ))
    db.commit()
    return summary


@app.get("/api/verification/sessions/{session_id}", tags=["verification"], summary="Получение данных конкретной сессии верификации (по идентификатору).", response_model=VerificationSessionDetailResponse)
def get_verification_session(session_id: UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = db.get(VerificationSession, str(session_id))
    if not session:
        raise HTTPException(404, "Сессия не найдена")
    return verification_session_payload(session)


@app.get(
    "/api/verification/sessions/{session_id}/status",
    tags=["verification"],
    summary="Получить статус и Ground Truth сессии верификации",
    response_model=VerificationSessionStatusResponse,
)
def get_verification_session_status(
    session_id: UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session_id_value = str(session_id)
    session = db.get(VerificationSession, session_id_value)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    review_task = db.scalar(
        select(ReviewTask)
        .where(ReviewTask.verification_session_id == session_id_value)
        .order_by(ReviewTask.created_at.desc())
        .limit(1)
    )
    return {
        "session_id": session.id,
        "status": session.status,
        "decision": session.decision,
        "is_match": session.is_match,
        "is_live": session.is_live,
        "ground_truth_match": session.ground_truth_match,
        "ground_truth_liveness": session.ground_truth_liveness,
        "review_status": review_task.status if review_task else None,
        "review_resolution": review_task.resolution if review_task else None,
        "reviewer_note": review_task.reviewer_note if review_task else None,
    }


@app.post(
    "/api/verification/sessions/{session_id}/complete", tags=["verification"],
    summary="Завершить приём видео для сессии биометрической верификации",
    response_model=VideoReceiptCompleteResponse,
)
def complete_verification_session_video(
    session_id: UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = db.get(VerificationSession, str(session_id))
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    details = dict(session.details or {})
    receipt = details.get("video_receipt") or {}
    completed_at = receipt.get("completed_at")
    if receipt.get("status") != "complete" or not completed_at:
        completed_at = datetime.now(timezone.utc).isoformat()
        details["video_receipt"] = {
            "status": "complete",
            "completed_at": completed_at,
        }
        session.details = details
        db.add(ActivityItem(
            type="verification",
            title="Приём видео завершён",
            description=f"Сессия {session.id}: видеозапись принята",
            status="success",
        ))
        db.commit()

    return {
        "session_id": session.id,
        "video_received": True,
        "completed_at": completed_at,
    }


@app.delete("/api/verification/sessions/{session_id}", tags=["verification"], status_code=204, summary="Удаление сессии биометрической верификации")
def delete_verification_session(session_id: UUID, db: Session = Depends(get_db)) -> Response:
    session_id_value = str(session_id)
    session = db.get(VerificationSession, session_id_value)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    db.execute(delete(ReviewTask).where(ReviewTask.verification_session_id == session_id_value))
    db.delete(session)
    db.add(ActivityItem(
        type="verification",
        title="Сессия верификации удалена",
        description=f"{session.employee_ref} · {session.audit_ref}",
        status="attention",
    ))
    db.commit()
    return Response(status_code=204)


@app.post(
    "/api/verification/sessions/{session_id}/review-request",
    tags=["verification"],
    response_model=ReviewTaskResponse,
    summary="Отправить автоматическое совпадение в очередь ручной проверки",
)
def request_verification_review(
    session_id: UUID,
    db: Session = Depends(get_db),
) -> ReviewTask:
    session = db.scalar(
        select(VerificationSession)
        .where(VerificationSession.id == str(session_id))
        .with_for_update()
    )
    if not session:
        raise HTTPException(404, "Сессия не найдена")
    if session.status != "verified" or session.decision != "match":
        raise HTTPException(
            409,
            "В очередь можно отправить только автоматически подтверждённое совпадение.",
        )

    active_task = db.scalar(
        select(ReviewTask)
        .where(
            ReviewTask.verification_session_id == session.id,
            ReviewTask.status.in_(["open", "in_review"]),
        )
        .order_by(ReviewTask.created_at.desc())
    )
    if active_task:
        sync_review_task_details(session, active_task, trigger="operator_request")
        db.commit()
        db.refresh(active_task)
        return active_task

    match_score = (
        f"{session.match_score:.0%}"
        if session.match_score is not None
        else "не оценено"
    )
    task = ReviewTask(
        kind="verification",
        subject_ref=session.employee_ref,
        status="open",
        priority="medium",
        summary=(
            "Ручная проверка по запросу оператора. "
            f"Автоматическое решение системы сохранено без изменений: "
            f"совпадение ({match_score})."
        ),
        verification_session_id=session.id,
    )
    db.add(task)
    db.flush()
    sync_review_task_details(session, task, trigger="operator_request")
    db.add(ActivityItem(
        type="review",
        title="Совпадение отправлено на ручную проверку",
        description=f"{session.employee_ref} · {session.audit_ref} · решение системы: match",
        status="attention",
    ))
    db.commit()
    db.refresh(task)
    return task


@app.get("/api/verification/metrics", tags=["verification"], response_model=list[BiometricMetricsResponse], summary="История метрик биометрической верификации")
def list_verification_metrics(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)) -> list[BiometricMetricsLog]:
    return list(db.scalars(select(BiometricMetricsLog).order_by(BiometricMetricsLog.created_at.desc()).limit(limit)))


@app.post("/api/verification/metrics/calculate", tags=["verification"], response_model=BiometricMetricsResponse, status_code=201, summary="Рассчитать и сохранить метрики биометрической верификации")
def calculate_verification_metrics(
    window_days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
) -> BiometricMetricsLog:
    record = BiometricMetricsLog(**calculate_metrics_snapshot(db, window_days))
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/api/review/tasks", tags=["review"], summary="Получение списка задач на проверку (review tasks)",  response_model=list[ReviewTaskResponse])
def list_review_tasks(status: str = Query("all"), limit: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)) -> list[ReviewTask]:
    stmt = select(ReviewTask).order_by(ReviewTask.created_at.desc()).limit(limit)
    if status != "all":
        stmt = stmt.where(ReviewTask.status == status)
    return list(db.scalars(stmt))


@app.post("/api/review/tasks/{task_id}/decision", tags=["review"], summary="Принятие решения по задаче на проверку (подтверждение/отклонение)", response_model=ReviewTaskResponse)
def resolve_review_task(task_id: UUID, payload: ReviewInput, db: Session = Depends(get_db)) -> ReviewTask:
    task = db.get(ReviewTask, str(task_id))
    if not task:
        raise HTTPException(404, "Задача не найдена")
    if task.kind == "verification" and payload.resolution != "request_retry":
        if payload.ground_truth_match is None or payload.ground_truth_liveness is None:
            raise HTTPException(422, "Для ручной разметки укажите, совпадает ли лицо и является ли оно живым.")
        if not task.verification_session_id:
            raise HTTPException(409, "У задачи не связана сессия верификации; разметку сохранить нельзя.")
        session = db.get(VerificationSession, task.verification_session_id)
        if session is None:
            raise HTTPException(409, "Связанная сессия верификации не найдена.")
        session.ground_truth_match = payload.ground_truth_match
        session.ground_truth_liveness = payload.ground_truth_liveness
    elif task.kind != "verification" and (
        payload.ground_truth_match is not None or payload.ground_truth_liveness is not None
    ):
        raise HTTPException(422, "Разметка биометрии применима только к задачам верификации.")
    task.status = "resolved"
    task.resolution = payload.resolution
    task.reviewer_note = payload.reviewer_note
    task.resolved_at = datetime.utcnow()
    if task.kind == "verification" and task.verification_session_id:
        linked_session = db.get(VerificationSession, task.verification_session_id)
        if linked_session:
            sync_review_task_details(linked_session, task, trigger="review_resolution")
    db.add(ActivityItem(type="review", title="Задача ручной проверки закрыта", description=f"{task.subject_ref}: {payload.resolution}", status="success"))
    db.commit()
    db.refresh(task)
    return task


@app.get(
    "/api/verification/review-queue",
    tags=["verification"],
    summary="Получить сессии, ожидающие ручной проверки",
    response_model=list[VerificationReviewQueueItemResponse],
)
def list_pending_verification_reviews(
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    rows = db.execute(
        select(ReviewTask, VerificationSession)
        .join(
            VerificationSession,
            ReviewTask.verification_session_id == VerificationSession.id,
        )
        .where(
            ReviewTask.kind == "verification",
            ReviewTask.status.in_(["open", "in_review"]),
        )
        .order_by(ReviewTask.created_at.asc())
    ).all()
    return [
        {
            "verification_session": {
                "id": session.id,
                "employee_ref": session.employee_ref,
                "status": session.status,
                "match_score": session.match_score,
                "liveness_score": session.liveness_score,
                "quality_score": session.quality_score,
                "decision": session.decision,
            },
        }
        for task, session in rows
    ]


@app.post(
    "/api/verification/sessions/{session_id}/review-decision",
    tags=["verification"],
    summary="Принять решение по ручной проверке сессии",
    response_model=ReviewTaskResponse,
)
def resolve_verification_session_review(
    session_id: UUID,
    payload: VerificationReviewDecisionInput,
    db: Session = Depends(get_db),
) -> ReviewTask:
    session_id_value = str(session_id)
    session = db.get(VerificationSession, session_id_value)
    if not session:
        raise HTTPException(404, "Сессия не найдена")

    task = db.scalar(
        select(ReviewTask)
        .where(
            ReviewTask.kind == "verification",
            ReviewTask.verification_session_id == session_id_value,
            ReviewTask.status.in_(["open", "in_review"]),
        )
        .order_by(ReviewTask.created_at.desc())
        .with_for_update()
    )
    if not task:
        raise HTTPException(409, "Сессия не ожидает ручной проверки.")

    return resolve_review_task(
        UUID(task.id),
        ReviewInput(
            resolution=payload.status,
            reviewer_note=payload.comment,
            ground_truth_match=payload.ground_truth_match,
            ground_truth_liveness=payload.ground_truth_liveness,
        ),
        db,
    )


@app.get("/api/governance/model-card", tags=["governance"], summary="Получение карточки модели (метаданные, характеристики, метрики)", response_model=ModelCard)
def get_model_card(db: Session = Depends(get_db)) -> ModelCard:
    model = active_model_or_503(db)
    return ModelCard(
        model_id=model.id,
        model_name=model.model_name,
        version=model.version,
        status=model.status,
        purpose=model.purpose,
        change_summary=model.change_summary,
        disclaimer=DISCLAIMER,
        thresholds={"stable": 0.72, "attention": 0.55, "completeness": 0.7},
        components=[
            ModelComponent(key="body", label="Физическое состояние", weight=0.4, source="витальные показатели"),
            ModelComponent(key="mental", label="Психоэмоциональная нагрузка", weight=0.35, source="сон, стресс, качество сигнала"),
            ModelComponent(key="social", label="Активность и восстановление", weight=0.25, source="активность, дыхание"),
        ],
        limitations=["Не является медицинским изделием.", "Не используется для постановки диагноза или допуска к работе.", "Требует клинической валидации R0–R4 и human review спорных случаев."],
        parameters=model.parameters,
        parameter_fields=parameter_schema(),
    )


@app.get("/api/governance/versions", summary="Получение списка версий моделей", tags=["governance"], response_model=list[ModelVersion])
def list_model_versions(db: Session = Depends(get_db)) -> list[ModelVersion]:
    models = list(db.scalars(select(RegisteredModel).order_by(RegisteredModel.created_at.desc())))
    versions = [
        ModelVersion(id=model.id, model_name=model.model_name, version=model.version, released_at=model.created_at, status=model.status, change_summary=model.change_summary)
        for model in models
    ]
    versions.append(ModelVersion(version="0.9.0-pilot", released_at=datetime(2026, 6, 18), status="archived", change_summary="Пилотный расчёт на ретроспективной выборке."))
    return versions


@app.get("/api/governance/models/defaults", summary="Системные значения модели", tags=["governance"])
def get_model_defaults() -> dict[str, Any]:
    return {
        "model_name": "HEALTH_ID",
        "purpose": "Версионированный расчёт исследовательского индекса и биометрической верификации.",
        "change_summary": "Новая версия модели.",
        "parameters": MODEL_CONFIG.public_settings(),
        "parameter_fields": parameter_schema(),
    }


@app.get("/api/governance/models", tags=["governance"], response_model=list[ModelRegistryResponse], summary="Список моделей")
def list_registered_models(db: Session = Depends(get_db)) -> list[ModelRegistryResponse]:
    return [model_payload(model) for model in db.scalars(select(RegisteredModel).order_by(RegisteredModel.created_at.desc()))]


@app.post("/api/governance/models", tags=["governance"], response_model=ModelRegistryResponse, status_code=201, summary="Добавить модель")
def create_registered_model(payload: ModelRegistryInput, db: Session = Depends(get_db)) -> ModelRegistryResponse:
    version = payload.version.strip()
    if not payload.model_name.strip() or not version or not payload.purpose.strip() or not payload.change_summary.strip():
        raise HTTPException(422, "Название, версия, назначение и описание не должны быть пустыми.")
    if db.scalar(select(RegisteredModel).where(RegisteredModel.version == version)):
        raise HTTPException(409, "Версия модели уже существует.")
    model = RegisteredModel(
        model_name=payload.model_name.strip(),
        version=version,
        status="draft",
        purpose=payload.purpose.strip(),
        change_summary=payload.change_summary.strip(),
        parameters=validated_model_parameters(payload.parameters),
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model_payload(model)


@app.put("/api/governance/models/{model_id}", tags=["governance"], response_model=ModelRegistryResponse, summary="Изменить модель")
def update_registered_model(model_id: str, payload: ModelRegistryInput, db: Session = Depends(get_db)) -> ModelRegistryResponse:
    model = db.get(RegisteredModel, model_id)
    if model is None:
        raise HTTPException(404, "Модель не найдена.")
    version = payload.version.strip()
    if not payload.model_name.strip() or not version or not payload.purpose.strip() or not payload.change_summary.strip():
        raise HTTPException(422, "Название, версия, назначение и описание не должны быть пустыми.")
    duplicate = db.scalar(select(RegisteredModel).where(RegisteredModel.version == version, RegisteredModel.id != model_id))
    if duplicate:
        raise HTTPException(409, "Версия модели уже существует.")
    model.model_name = payload.model_name.strip()
    model.version = version
    model.purpose = payload.purpose.strip()
    model.change_summary = payload.change_summary.strip()
    model.parameters = validated_model_parameters(payload.parameters)
    db.commit()
    db.refresh(model)
    return model_payload(model)


@app.post("/api/governance/models/{model_id}/activate", tags=["governance"], response_model=ModelRegistryResponse, summary="Активировать модель")
def activate_registered_model(model_id: str, db: Session = Depends(get_db)) -> ModelRegistryResponse:
    model = db.get(RegisteredModel, model_id)
    if model is None:
        raise HTTPException(404, "Модель не найдена.")
    for current in db.scalars(select(RegisteredModel).where(RegisteredModel.status == "current")):
        current.status = "draft"
    model.status = "current"
    db.commit()
    db.refresh(model)
    return model_payload(model)


@app.post("/api/governance/models/{model_id}/archive", tags=["governance"], response_model=ModelRegistryResponse, summary="Архивировать модель")
def archive_registered_model(model_id: str, db: Session = Depends(get_db)) -> ModelRegistryResponse:
    model = db.get(RegisteredModel, model_id)
    if model is None:
        raise HTTPException(404, "Модель не найдена.")
    if model.status == "current":
        raise HTTPException(409, "Сначала активируйте другую модель.")
    model.status = "archived"
    db.commit()
    db.refresh(model)
    return model_payload(model)