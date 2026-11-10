from __future__ import annotations

from contextlib import suppress
import math
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import HTTPException

from .verification_config import MODEL_CONFIG, ModelConfig


DISCLAIMER = (
    "Исследовательская оценка с настраиваемыми методами детекции и извлечения биометрических признаков. "
    "Не является юридической идентификацией или медицинским заключением; "
    "спорные результаты требуют проверки специалистом."
)
ALGORITHM_IMPLEMENTATION_VERSION = "ukz-biometric-v7"

cv2.setNumThreads(1)
_CASCADE_DIR = Path(cv2.data.haarcascades)


class InsufficientReferenceQualityError(ValueError):
    """Raised when a readable reference photo does not provide a usable face."""


def _load_cascade(filename: str) -> cv2.CascadeClassifier | None:
    path = _CASCADE_DIR / filename
    if not path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


_FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")
_FACE_CASCADE_ALT = _load_cascade("haarcascade_frontalface_alt.xml")
_FACE_CASCADE_ALT2 = _load_cascade("haarcascade_frontalface_alt2.xml")
_PROFILE_FACE_CASCADE = _load_cascade("haarcascade_profileface.xml")
_FACE_CASCADES = tuple(
    cascade
    for cascade in (_FACE_CASCADE, _FACE_CASCADE_ALT, _FACE_CASCADE_ALT2)
    if cascade is not None
)
_EYE_FILE = (
    "haarcascade_eye_tree_eyeglasses.xml"
    if (_CASCADE_DIR / "haarcascade_eye_tree_eyeglasses.xml").exists()
    else "haarcascade_eye.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(str(_CASCADE_DIR / _EYE_FILE))
_DEEPFACE: Any | None = None
_DEEPFACE_IMPORT_LOCK = Lock()
_DEEPFACE_BACKENDS = {
    "mtcnn": "mtcnn",
    "retinaface": "retinaface",
}


def _load_siamese_runtime(
    config: ModelConfig,
) -> tuple[Any | None, dict[str, str | None]]:
    app_dir = Path(__file__).resolve().parent
    models_dir = (app_dir / "models").resolve()
    configured_path = Path(config.liveness_siamese_model_path)
    checkpoint_path = (
        configured_path if configured_path.is_absolute() else app_dir / configured_path
    ).resolve()

    try:
        checkpoint_path.relative_to(models_dir)
    except ValueError:
        return None, {
            "status": "unavailable",
            "reason": "Путь к Siamese-модели должен указывать на файл внутри каталога моделей API.",
        }

    if not checkpoint_path.is_file():
        return None, {
            "status": "unavailable",
            "reason": f"Файл весов Siamese не найден: {checkpoint_path.name}.",
        }

    try:
        from .siamese_liveness import load_siamese_model

        return load_siamese_model(checkpoint_path), {
            "status": "available",
            "reason": None,
        }
    except ModuleNotFoundError as error:
        if error.name == "torch":
            return None, {
                "status": "unavailable",
                "reason": "Файл весов найден, но PyTorch не установлен; загрузить модель нельзя.",
            }
        return None, {
            "status": "unavailable",
            "reason": "Не удалось загрузить Siamese-модель из-за отсутствующей зависимости.",
        }
    except Exception:
        return None, {
            "status": "unavailable",
            "reason": (
                "Файл весов найден, но не соответствует архитектуре Siamese "
                "или не может быть загружен."
            ),
        }


def _assert_cascades_ready(require_face_cascade: bool = True) -> None:
    if _EYE_CASCADE.empty() or (require_face_cascade and not _FACE_CASCADES):
        raise RuntimeError("Required OpenCV Haar cascade files are unavailable")


def _load_deepface() -> Any:
    global _DEEPFACE
    if _DEEPFACE is None:
        with _DEEPFACE_IMPORT_LOCK:
            if _DEEPFACE is None:
                try:
                    from deepface import DeepFace
                except Exception as error:
                    raise HTTPException(
                        status_code=503,
                        detail="Не удалось загрузить локальный ML-модуль DeepFace. Проверьте зависимости биометрических методов.",
                    ) from error
                _DEEPFACE = DeepFace
    return _DEEPFACE


def _inference_backend_unavailable(method: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            f"Не удалось выполнить метод «{method}». Проверьте локальные зависимости "
            "и доступность загрузки весов модели при первом запуске."
        ),
    )


def _face_box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union else 0.0


def _deduplicate_face_boxes(
    boxes: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.38,
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if any(_face_box_iou(box, existing) >= overlap_threshold for existing in kept):
            continue
        kept.append(box)
    return kept


def _run_haar(
    cascade: cv2.CascadeClassifier | None,
    gray: np.ndarray,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    min_size: tuple[int, int] = (24, 24),
    upscale: float = 1.0,
    flip: bool = False,
) -> list[tuple[int, int, int, int]]:
    if cascade is None or cascade.empty() or gray.size == 0:
        return []

    source = cv2.flip(gray, 1) if flip else gray
    original_height, original_width = source.shape[:2]
    if upscale != 1.0:
        scaled_width = max(1, int(round(original_width * upscale)))
        scaled_height = max(1, int(round(original_height * upscale)))
        source = cv2.resize(source, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        scale_x = scaled_width / original_width
        scale_y = scaled_height / original_height
    else:
        scale_x = scale_y = 1.0

    try:
        detected = cascade.detectMultiScale(
            source,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
    except cv2.error:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for raw_x, raw_y, raw_width, raw_height in detected:
        x, y = int(raw_x), int(raw_y)
        width, height = int(raw_width), int(raw_height)
        if flip:
            x = source.shape[1] - (x + width)
        x = int(round(x / scale_x))
        y = int(round(y / scale_y))
        width = int(round(width / scale_x))
        height = int(round(height / scale_y))
        if width < 24 or height < 24:
            continue
        aspect_ratio = width / height
        if not 0.55 <= aspect_ratio <= 1.55:
            continue
        boxes.append((x, y, width, height))
    return boxes


def _consensus_face_boxes(
    detections: list[tuple[str, tuple[int, int, int, int]]],
) -> list[tuple[int, int, int, int]]:
    clusters: list[dict[str, Any]] = []
    for source, box in detections:
        matching = [
            cluster for cluster in clusters
            if _face_box_iou(box, cluster["anchor"]) >= 0.25
        ]
        if matching:
            cluster = max(matching, key=lambda item: _face_box_iou(box, item["anchor"]))
            cluster["boxes"].append(box)
            cluster["sources"].add(source)
        else:
            clusters.append({"anchor": box, "boxes": [box], "sources": {source}})

    consensus: list[tuple[int, int, int, int]] = []
    for cluster in clusters:
        if len(cluster["sources"]) < 2:
            continue
        boxes = cluster["boxes"]
        consensus.append(tuple(
            int(round(sum(box[index] for box in boxes) / len(boxes)))
            for index in range(4)
        ))
    return _deduplicate_face_boxes(consensus)


def _detect_faces(
    image: np.ndarray,
    backend: str = "haar",
    *,
    robust: bool = True,
) -> list[tuple[int, int, int, int]]:
    if backend == "haar":
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray)

        # Preserve the fast path for video frames; search the alternate cascades
        # and normalized versions for reference portraits or explicit retries.
        primary_cascade = _FACE_CASCADE or next(iter(_FACE_CASCADES), None)
        primary_boxes = _run_haar(
            primary_cascade,
            gray,
            scale_factor=1.1,
            min_neighbors=4,
            min_size=(24, 24),
        )
        if primary_boxes:
            return _deduplicate_face_boxes(primary_boxes)
        if not robust:
            return []

        strict_boxes: list[tuple[int, int, int, int]] = []
        for cascade in _FACE_CASCADES:
            strict_boxes.extend(_run_haar(cascade, gray, min_neighbors=4, min_size=(24, 24)))
        if strict_boxes:
            return _deduplicate_face_boxes(strict_boxes)

        profile_boxes = _run_haar(
            _PROFILE_FACE_CASCADE,
            gray,
            min_neighbors=4,
            min_size=(24, 24),
        )
        profile_boxes.extend(_run_haar(
            _PROFILE_FACE_CASCADE,
            gray,
            min_neighbors=4,
            min_size=(24, 24),
            flip=True,
        ))
        if profile_boxes:
            return _deduplicate_face_boxes(profile_boxes)

        normalized: list[tuple[str, np.ndarray]] = [
            ("equalized", cv2.equalizeHist(gray)),
            ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)),
        ]
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 80 or mean_brightness > 180:
            gamma = 0.7 if mean_brightness < 80 else 1.3
            table = np.clip(
                ((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            normalized.append(("gamma", cv2.LUT(gray, table)))

        detections: list[tuple[str, tuple[int, int, int, int]]] = []
        min_edge = min(gray.shape[:2])
        upscale = min(2.0, 480.0 / min_edge) if min_edge and min_edge < 480 else 1.0
        for variant_name, variant in normalized:
            for cascade_index, cascade in enumerate(_FACE_CASCADES):
                source_name = f"{variant_name}:{cascade_index}"
                boxes = _run_haar(
                    cascade,
                    variant,
                    scale_factor=1.08,
                    min_neighbors=3,
                    min_size=(24, 24),
                )
                detections.extend((source_name, box) for box in boxes)
                if upscale > 1.0:
                    enlarged_boxes = _run_haar(
                        cascade,
                        variant,
                        scale_factor=1.08,
                        min_neighbors=3,
                        min_size=(30, 30),
                        upscale=upscale,
                    )
                    detections.extend((f"{source_name}:upscaled", box) for box in enlarged_boxes)
        return _consensus_face_boxes(detections)

    deepface_backend = _DEEPFACE_BACKENDS.get(backend)
    if deepface_backend is None:
        raise HTTPException(status_code=422, detail=f"Неизвестный детектор лица: «{backend}».")

    try:
        detections = _load_deepface().extract_faces(
            img_path=image,
            detector_backend=deepface_backend,
            enforce_detection=False,
            align=False,
            normalize_face=False,
            color_face="bgr",
        )
    except HTTPException:
        raise
    except Exception as error:
        raise _inference_backend_unavailable(backend) from error

    image_height, image_width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for detection in detections or []:
        if float(detection.get("confidence") or 0) <= 0:
            continue
        area = detection.get("facial_area") or {}
        x = max(0, int(area.get("x", 0)))
        y = max(0, int(area.get("y", 0)))
        width = min(int(area.get("w", 0)), image_width - x)
        height = min(int(area.get("h", 0)), image_height - y)
        if width >= 48 and height >= 48:
            boxes.append((x, y, width, height))
    return boxes


def _crop_face(gray: np.ndarray, box: tuple[int, int, int, int], padding: float = 0.08) -> np.ndarray:
    x, y, width, height = box
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(gray.shape[1], x + width + pad_x)
    bottom = min(gray.shape[0], y + height + pad_y)
    return gray[top:bottom, left:right]


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    center = small[1:-1, 1:-1]
    neighbors = (
        small[:-2, :-2],
        small[:-2, 1:-1],
        small[:-2, 2:],
        small[1:-1, 2:],
        small[2:, 2:],
        small[2:, 1:-1],
        small[2:, :-2],
        small[1:-1, :-2],
    )
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    histogram = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    total = float(histogram.sum())
    return histogram / total if total else histogram


def _matching_lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """LBP used by the HOG-LBP matcher; keep liveness texture features separate."""
    small = cv2.resize(gray, (64, 64))
    padded = cv2.copyMakeBorder(small, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    codes = np.zeros(small.shape, dtype=np.uint8)
    for bit in range(8):
        angle = 2 * np.pi * bit / 8
        dx = int(round(np.cos(angle)))
        dy = int(round(np.sin(angle)))
        neighbor = padded[1 + dy:1 + dy + small.shape[0], 1 + dx:1 + dx + small.shape[1]]
        codes |= ((neighbor >= small).astype(np.uint8) << bit)
    histogram = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    total = float(histogram.sum())
    return histogram / total if total else histogram


def _embedding(
    face_image: np.ndarray,
    method: str = "hog_lbp",
    model_embedding_callback: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    face_gray = (
        face_image
        if face_image.ndim == 2
        else cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    )
    if method == "resnet50_arcface":
        if model_embedding_callback is None:
            raise HTTPException(
                status_code=503,
                detail="Обученная модель ResNet50 + ArcFace недоступна.",
            )
        face_bgr = (
            face_image
            if face_image.ndim == 3
            else cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
        )
        try:
            embedding = np.asarray(
                model_embedding_callback(face_bgr),
                dtype=np.float32,
            ).reshape(-1)
        except HTTPException:
            raise
        except Exception as error:
            raise _inference_backend_unavailable("ResNet50 + ArcFace") from error
        norm = float(np.linalg.norm(embedding))
        if not embedding.size or not np.isfinite(embedding).all() or norm <= 1e-12:
            raise HTTPException(
                status_code=503,
                detail="ResNet50 + ArcFace вернула некорректный вектор признаков.",
            )
        return np.ascontiguousarray(embedding / norm, dtype=np.float32)
    if method == "arcface":
        face_bgr = (
            face_image
            if face_image.ndim == 3
            else cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
        )
        try:
            representations = _load_deepface().represent(
                img_path=face_bgr,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip",
                align=False,
                normalization="ArcFace",
                l2_normalize=True,
            )
            embedding = np.asarray(representations[0]["embedding"], dtype=np.float32)
        except HTTPException:
            raise
        except Exception as error:
            raise _inference_backend_unavailable("ArcFace") from error
        norm = float(np.linalg.norm(embedding))
        if not embedding.size or not np.isfinite(embedding).all() or norm == 0:
            raise HTTPException(status_code=503, detail="ArcFace вернул некорректный вектор признаков.")
        return embedding / norm
    if method != "hog_lbp":
        raise HTTPException(status_code=422, detail=f"Неизвестный метод представления лица: «{method}».")

    normalized = cv2.resize(face_gray, (64, 64))
    hog = cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    ).compute(normalized).reshape(-1).astype(np.float32)
    lbp = _matching_lbp_histogram(normalized)
    descriptor = np.concatenate((hog, lbp)).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    return descriptor / norm if norm else descriptor


def _raw_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return -1.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _cosine_score(cosine: float) -> float:
    """Map cosine similarity [-1, 1] onto the reference implementation's [0, 1] scale."""
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return _cosine_score(_raw_cosine_similarity(first, second))


def _eye_boxes(face_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    upper = face_gray[: max(1, int(face_gray.shape[0] * 0.65)), :]
    boxes = _EYE_CASCADE.detectMultiScale(
        upper,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(8, 8),
    )
    candidates = [tuple(int(value) for value in box) for box in boxes]
    candidates.sort(key=lambda item: (item[1], -item[2] * item[3]))
    return candidates[:4]


def _estimate_pose(
    box: tuple[int, int, int, int],
    face_gray: np.ndarray,
    eyes: list[tuple[int, int, int, int]],
    frame_shape: tuple[int, ...],
) -> dict[str, float] | None:
    if len(eyes) < 2:
        return None
    eye_pair = sorted(eyes, key=lambda item: item[0])[:2]
    left_x = eye_pair[0][0] + eye_pair[0][2] / 2
    right_x = eye_pair[1][0] + eye_pair[1][2] / 2
    eye_y1 = eye_pair[0][1] + eye_pair[0][3] / 2
    eye_y2 = eye_pair[1][1] + eye_pair[1][3] / 2
    face_height, face_width = face_gray.shape[:2]

    left_distance = max(1.0, left_x)
    right_distance = max(1.0, face_width - right_x)
    asymmetry = (left_distance - right_distance) / (left_distance + right_distance)
    yaw = float(np.clip(asymmetry * 90.0, -45.0, 45.0))
    pitch = float(np.clip(((((eye_y1 + eye_y2) / 2) / face_height) - 0.38) * 100, -40, 40))
    roll = math.degrees(math.atan2(eye_y2 - eye_y1, max(1.0, right_x - left_x)))
    frame_height, frame_width = frame_shape[:2]
    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": float(np.clip(roll, -45.0, 45.0)),
        "center_x": (box[0] + box[2] / 2) / max(1, frame_width),
        "center_y": (box[1] + box[3] / 2) / max(1, frame_height),
    }


def _texture_metrics(face_gray: np.ndarray, config: ModelConfig) -> dict[str, float | bool]:
    lbp = _lbp_histogram(face_gray)
    probabilities = lbp.astype(np.float64)
    nonzero = probabilities[probabilities > 0]
    entropy = float(np.clip(-np.sum(nonzero * np.log2(nonzero)) / 8.0, 0.0, 1.0))

    resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(resized - resized.mean()))) ** 2
    yy, xx = np.ogrid[:64, :64]
    radius = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
    high_frequency = spectrum[radius >= 6]
    total = float(high_frequency.sum())
    peak_ratio = float(high_frequency.max() / total) if total > 0 else 0.0
    fft_suspicious = peak_ratio > config.texture_freq_threshold
    return {
        "lbp_entropy": entropy,
        "fft_peak_ratio": peak_ratio,
        "fft_suspicious": bool(fft_suspicious),
    }


def _band_peak(signal: np.ndarray, sample_rate: float, band: tuple[float, float]) -> dict[str, float | None]:
    signal = signal.astype(np.float64)
    if len(signal) < 4 or sample_rate <= 0:
        return {"frequency": None, "confidence": 0.0}
    signal -= np.linspace(signal[0], signal[-1], len(signal))
    signal -= signal.mean()
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
    indices = np.flatnonzero(band_mask)
    if len(indices) < 2:
        return {"frequency": None, "confidence": 0.0}
    peak_index = int(indices[np.argmax(spectrum[indices])])
    peak_power = float(spectrum[peak_index])
    neighborhood = (np.abs(frequencies - frequencies[peak_index]) > 0.12) & band_mask
    noise = float(np.median(spectrum[neighborhood])) if np.any(neighborhood) else 0.0
    snr = peak_power / max(noise, 1e-12)
    confidence = float(np.clip((math.log10(max(snr, 1.0)) - 0.1) / 1.8, 0.0, 1.0))
    return {"frequency": float(frequencies[peak_index]), "confidence": confidence}


def _estimate_rppg(
    green_samples: list[float],
    fps: float,
    config: ModelConfig,
) -> dict[str, Any]:
    duration = len(green_samples) / fps if fps > 0 else 0.0
    base = {
        "heart_rate": None,
        "breathing_rate": None,
        "hrv": None,
        "signal_quality": 0.0,
        "confidence": 0.0,
        "duration_sec": round(duration, 2),
    }
    if fps < config.min_fps or duration < config.rppg_min_duration_sec:
        return {
            **base,
            "status": "unavailable",
            "reason": f"Для rPPG нужно не менее {config.rppg_min_duration_sec:g} секунд при {config.min_fps:g} fps.",
        }

    values = np.asarray(green_samples, dtype=np.float64)
    effective_fps = min(fps, config.rppg_target_fps)
    if fps > effective_fps:
        target_count = int(duration * effective_fps)
        old_times = np.arange(len(values), dtype=np.float64) / fps
        new_times = np.arange(target_count, dtype=np.float64) / effective_fps
        values = np.interp(new_times, old_times, values)

    heart = _band_peak(values, effective_fps, config.rppg_heart_rate_band)
    breathing = _band_peak(values, effective_fps, config.rppg_breathing_band)
    confidence = float(min(heart["confidence"], breathing["confidence"]))
    signal_quality = float((heart["confidence"] + breathing["confidence"]) / 2)

    hrv_ms: float | None = None
    if heart["frequency"] is not None:
        detrended = values - np.linspace(values[0], values[-1], len(values))
        filtered_spectrum = np.fft.rfft(detrended)
        frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / effective_fps)
        mask = (frequencies >= config.rppg_heart_rate_band[0]) & (frequencies <= config.rppg_heart_rate_band[1])
        pulse = np.fft.irfft(filtered_spectrum * mask, n=len(detrended))
        minimum_gap = max(1, int(effective_fps * 0.33))
        threshold = float(pulse.mean() + 0.25 * pulse.std())
        peaks: list[int] = []
        for index in range(1, len(pulse) - 1):
            if (
                pulse[index] > threshold
                and pulse[index] >= pulse[index - 1]
                and pulse[index] > pulse[index + 1]
                and (not peaks or index - peaks[-1] >= minimum_gap)
            ):
                peaks.append(index)
        if len(peaks) >= 4:
            intervals = np.diff(peaks) / effective_fps
            hrv_ms = float(np.clip(np.std(intervals) * 1000, 0, 300))

    status = "available" if confidence >= config.liveness_rppg_min_confidence else "low_confidence"
    return {
        **base,
        "heart_rate": round(float(heart["frequency"]) * 60, 1) if heart["frequency"] else None,
        "breathing_rate": round(float(breathing["frequency"]) * 60, 1) if breathing["frequency"] else None,
        "hrv": round(hrv_ms, 1) if hrv_ms is not None else None,
        "signal_quality": round(signal_quality, 4),
        "confidence": round(confidence, 4),
        "status": status,
        "reason": None if status == "available" else "Качество rPPG-сигнала ниже настроенного порога.",
    }


def _score_liveness(
    blink_score: float,
    motion_score: float,
    texture_score: float,
    rppg: dict[str, Any],
    config: ModelConfig,
    siamese_score: float | None,
    siamese_available: bool,
) -> tuple[float, str, str | None]:
    if config.liveness_method == "siamese":
        if not siamese_available:
            return 0.5, "inconclusive", "configured_model_unavailable"
        if siamese_score is None:
            return 0.5, "inconclusive", None
        if siamese_score >= config.liveness_siamese_threshold:
            return round(float(siamese_score), 4), "live", None
        return (
            round(float(siamese_score), 4),
            "spoof_detected",
            "presentation_attack_suspected",
        )
    if config.liveness_method not in {"ensemble", "rppg"}:
        return 0.5, "inconclusive", "configured_model_unavailable"

    if config.liveness_method == "rppg":
        score = float(rppg["confidence"]) if rppg["status"] == "available" else 0.5
    else:
        if not siamese_available:
            return 0.5, "inconclusive", "configured_model_unavailable"
        if siamese_score is None:
            return 0.5, "inconclusive", None
        components = [
            (siamese_score, config.liveness_siamese_weight),
            (blink_score, config.blink_weight),
            (motion_score, config.head_motion_weight),
            (texture_score, config.texture_weight),
        ]
        if rppg["status"] == "available":
            components.append((float(rppg["confidence"]), config.liveness_rppg_weight))
        if config.depth_enabled:
            # No calibrated depth camera is available in this webcam upload flow.
            # Exclude depth rather than claim a depth measurement.
            pass

        values = [value for value, _ in components]
        weights = [weight for _, weight in components]
        if config.liveness_strategy == "and":
            score = min(values)
        elif config.liveness_strategy == "or":
            score = max(values)
        elif config.liveness_strategy in {"weighted_avg", "adaptive"}:
            score = sum(value * weight for value, weight in components) / max(sum(weights), 1e-9)
        else:
            return 0.5, "inconclusive", "configured_model_unavailable"

    if score >= config.liveness_pass:
        return round(float(score), 4), "live", None
    if score < config.liveness_fail:
        return round(float(score), 4), "spoof_detected", "presentation_attack_suspected"
    return round(float(score), 4), "inconclusive", None


def _select_face_candidate(
    frame: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]],
    reference_embedding: np.ndarray,
    embedding_method: str,
    strategy: str,
    max_faces_per_frame: int,
    padding: float,
    model_embedding_callback: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, float] | None:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy in {"skip", "sckip"}:
        normalized_strategy = "skip_frame"
    if normalized_strategy == "skip_frame" and len(face_boxes) > max_faces_per_frame:
        return None

    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for box in face_boxes:
        crop = _crop_face(frame, box, padding=padding)
        if crop.size:
            candidates.append((box, crop))
    if not candidates:
        return None

    if normalized_strategy in {"max", "skip_frame"}:
        selected_box, selected_crop = max(
            candidates,
            key=lambda candidate: candidate[0][2] * candidate[0][3],
        )
        descriptor = _embedding(
            selected_crop, embedding_method, model_embedding_callback
        )
        raw_score = _raw_cosine_similarity(reference_embedding, descriptor)
        return selected_box, selected_crop, descriptor, raw_score

    if normalized_strategy != "select_closest_to_reference":
        raise ValueError(f"Unsupported multi-face strategy: {strategy}")

    scored_candidates: list[
        tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, float]
    ] = []
    for box, crop in candidates:
        descriptor = _embedding(crop, embedding_method, model_embedding_callback)
        raw_score = _raw_cosine_similarity(reference_embedding, descriptor)
        scored_candidates.append((box, crop, descriptor, raw_score))
    return max(scored_candidates, key=lambda candidate: candidate[3])


def build_not_evaluable_result(
    reason: str,
    config: ModelConfig,
    model_version: str,
) -> dict[str, Any]:
    """Build a persisted, reviewable result when the pipeline cannot produce scores."""
    clean_reason = " ".join(str(reason).split())[:500] or "Системе не удалось выполнить достоверную оценку."
    details = {
        "event_id": "",
        "model_version": model_version,
        "implementation_version": ALGORITHM_IMPLEMENTATION_VERSION,
        "config_hash": config.fingerprint(),
        "face_match": {
            "match_score": None,
            "per_frame_scores": [],
            "frame_ids": [],
            "best_frame_id": None,
            "threshold": config.match_threshold,
            "manual_review_threshold": config.manual_review_threshold,
            "decision": "not_evaluable",
            "total_frames": 0,
            "detected_frames": 0,
            "detector_backend": config.detector_backend,
            "embedding_method": config.embedding_method,
        },
        "liveness": {
            "liveness_score": None,
            "decision": "not_evaluable",
            "method": config.liveness_method,
            "model_status": {
                "status": "not_evaluated",
                "reason": clean_reason,
            },
            "liveness_signals": {},
            "attack_type": None,
        },
        "quality": {
            "overall": "not_evaluable",
            "checks": {},
            "blocking_issues": [clean_reason],
            "warnings": [],
        },
        "route": "not_evaluable",
        "not_evaluable_reason": clean_reason,
        "recommended_action": "retry_or_manual_review",
        "health_id": {
            "status": "not_calculated",
            "health_id_value": None,
            "completeness": 0.0,
            "uncertainty_score": 1.0,
            "category": "review",
            "evidence": {},
            "error": clean_reason,
        },
        "disclaimer": DISCLAIMER,
    }
    return {
        "status": "not_evaluable",
        "match_score": None,
        "liveness_score": None,
        "quality_score": None,
        "decision": "review",
        "attack_type": None,
        "details": details,
    }


def build_insufficient_quality_result(
    reason: str,
    config: ModelConfig,
    model_version: str,
) -> dict[str, Any]:
    """Build a reviewable quality result when input files are readable but unusable."""
    clean_reason = " ".join(str(reason).split())[:500] or "Качество исходных данных недостаточно."
    result = build_not_evaluable_result(clean_reason, config, model_version)
    details = result["details"]
    details.pop("not_evaluable_reason", None)
    details["route"] = "insufficient_quality"
    details["quality"] = {
        "overall": "reject",
        "checks": {
            "reference_face": {
                "status": "fail",
                "actual": 0,
                "required": 1,
                "note": clean_reason,
            },
        },
        "blocking_issues": [clean_reason],
        "warnings": [],
    }
    details["face_match"]["decision"] = "insufficient_quality"
    details["liveness"]["decision"] = "insufficient_quality"
    details["liveness"]["model_status"] = {
        "status": "not_evaluated",
        "reason": clean_reason,
    }
    details["recommended_action"] = "retry_or_manual_review"
    return {
        **result,
        "status": "insufficient_quality",
        "quality_score": 0.0,
        "details": details,
    }


def _classify_multi_face_candidates(
    candidate_frames: int,
    sampled_frames: int,
    warn_threshold: float,
    block_threshold: float,
) -> tuple[float, str, bool]:
    ratio = candidate_frames / max(1, sampled_frames)
    if candidate_frames <= 0:
        return ratio, "normal", False
    if ratio > block_threshold:
        return ratio, "high", True
    if ratio >= warn_threshold:
        return ratio, "elevated", False
    return ratio, "normal", False


def _resolve_verification_status(
    *,
    liveness_component_unavailable: bool,
    blocking_issues: list[str],
    match_decision: str,
    match_score: float,
    liveness_decision: str,
    quality_score: float,
    quality_threshold: float,
    manual_review_threshold: float,
) -> str:
    if liveness_component_unavailable:
        return "not_evaluable"
    if blocking_issues or quality_score < quality_threshold:
        return "insufficient_quality"
    if match_decision == "match" and liveness_decision == "live":
        return "verified"
    if match_score >= manual_review_threshold or liveness_decision == "inconclusive":
        return "manual_review"
    return "not_verified"


def capture_quality_report(
    width: int,
    height: int,
    fps: float,
    config: ModelConfig,
) -> dict[str, Any]:
    min_width, min_height = config.min_resolution
    resolution_ok = width >= min_width and height >= min_height
    fps_ok = fps >= config.min_fps
    fps_tolerance = config.min_fps * 0.01
    fps_acceptable = fps >= config.min_fps - fps_tolerance
    return {
        "resolution_ok": resolution_ok,
        "fps_ok": fps_ok,
        "fps_acceptable": fps_acceptable,
        "resolution": {
            "status": "pass" if resolution_ok else "fail",
            "actual": [width, height],
            "required": [min_width, min_height],
        },
        "fps": {
            "status": "pass" if fps_ok else "warning" if fps_acceptable else "fail",
            "actual": round(fps, 2),
            "required": config.min_fps,
            "tolerance": round(fps_tolerance, 2),
        },
    }


def _uniform_frame_indices(total_frames: int, desired_samples: int) -> list[int]:
    if total_frames <= 0 or desired_samples <= 0:
        return []
    sample_count = min(total_frames, desired_samples)
    indices = np.rint(
        np.linspace(0, total_frames - 1, num=sample_count)
    ).astype(np.int64)
    return [int(index) for index in np.unique(indices)]


def _evaluate_frame_quality(
    gray_crop: np.ndarray,
    area_pct: float,
    pose: dict[str, float] | None,
    config: ModelConfig,
) -> dict[str, Any]:
    if gray_crop.size == 0:
        return {
            "score": 0.0,
            "eligible": False,
            "reasons": ["empty_face_crop"],
            "sharpness": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "illumination_spread": 0.0,
            "face_area_pct": round(float(area_pct), 3),
        }

    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    brightness = float(np.mean(gray_crop))
    contrast = float(np.std(gray_crop))
    sigma = max(1.0, min(gray_crop.shape[:2]) * 0.12)
    low_frequency = cv2.GaussianBlur(gray_crop, (0, 0), sigmaX=sigma)
    illumination_spread = float(
        np.percentile(low_frequency, 90) - np.percentile(low_frequency, 10)
    )

    reasons: list[str] = []
    if sharpness < config.max_blur_variance:
        reasons.append("blur")
    if brightness < config.frame_quality_min_brightness:
        reasons.append("underexposure")
    elif brightness > config.frame_quality_max_brightness:
        reasons.append("overexposure")
    if contrast < config.frame_quality_min_contrast:
        reasons.append("low_contrast")
    if illumination_spread > config.frame_quality_max_illumination_spread:
        reasons.append("uneven_illumination")
    if area_pct < config.min_face_area_pct:
        reasons.append("face_too_small")
    if pose and (
        abs(pose["yaw"]) > config.max_yaw
        or abs(pose["pitch"]) > config.max_pitch
        or abs(pose["roll"]) > config.max_roll
    ):
        reasons.append("pose_out_of_range")

    brightness_midpoint = (
        config.frame_quality_min_brightness + config.frame_quality_max_brightness
    ) / 2
    brightness_half_range = max(
        1.0,
        (config.frame_quality_max_brightness - config.frame_quality_min_brightness) / 2,
    )
    quality_components = [
        (
            min(1.0, sharpness / max(config.max_blur_variance * 3, 1.0)),
            0.30,
        ),
        (
            max(0.0, 1.0 - abs(brightness - brightness_midpoint) / brightness_half_range),
            0.20,
        ),
        (
            min(
                1.0,
                contrast / max(config.frame_quality_min_contrast * 3, 1.0),
            ),
            0.15,
        ),
        (
            max(
                0.0,
                1.0
                - illumination_spread
                / max(config.frame_quality_max_illumination_spread, 1.0),
            ),
            0.20,
        ),
        (
            min(1.0, area_pct / max(config.min_face_area_pct * 2, 1.0)),
            0.15,
        ),
    ]
    quality_score = sum(value * weight for value, weight in quality_components) / sum(
        weight for _, weight in quality_components
    )
    return {
        "score": round(float(quality_score), 6),
        "eligible": not reasons,
        "reasons": reasons,
        "sharpness": round(sharpness, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "illumination_spread": round(illumination_spread, 3),
        "face_area_pct": round(float(area_pct), 3),
    }


def _select_aggregation_frames(
    observations: list[dict[str, Any]],
    config: ModelConfig,
) -> list[dict[str, Any]]:
    eligible = [
        item for item in observations
        if item["frame_quality"]["eligible"]
    ]
    if config.frame_selection_strategy == "all_accepted_median":
        return sorted(eligible, key=lambda item: item["frame_id"])
    if config.frame_selection_strategy != "best_quality_median":
        raise ValueError(
            f"Unsupported frame selection strategy: {config.frame_selection_strategy}"
        )
    ranked = sorted(
        eligible,
        key=lambda item: (
            item["frame_quality"]["score"],
            item["sharpness"],
            -item["frame_id"],
        ),
        reverse=True,
    )
    selected = ranked[:config.frame_median_count]
    return sorted(selected, key=lambda item: item["frame_id"])


def analyze_biometric_video(
    photo_bytes: bytes,
    video_bytes: bytes,
    video_extension: str,
    config: ModelConfig | None = None,
    model_version: str = "face_v1.0.0",
    arcface_embedding_callback: Callable[[np.ndarray], np.ndarray] | None = None,
    model_embedding_callback: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute research-only face, liveness and capture-quality measurements."""
    config = config or MODEL_CONFIG
    _assert_cascades_ready(require_face_cascade=config.detector_backend == "haar")
    if not photo_bytes:
        raise HTTPException(status_code=422, detail="Эталонное фото не содержит данных.")
    try:
        reference = cv2.imdecode(np.frombuffer(photo_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise HTTPException(status_code=422, detail="Не удалось прочитать эталонное фото.") from error
    if reference is None:
        raise HTTPException(status_code=422, detail="Не удалось прочитать эталонное фото.")
    reference_faces = _detect_faces(reference, config.detector_backend, robust=True)
    if not reference_faces:
        raise InsufficientReferenceQualityError(
            "На читаемом эталонном фото не обнаружено лицо. Используйте чёткий снимок анфас "
            "с одним хорошо освещённым лицом без маски и сильного поворота."
        )
    reference_box = max(reference_faces, key=lambda item: item[2] * item[3])
    face_crop_padding = 0.0 if config.embedding_method == "hog_lbp" else 0.08
    reference_face = _crop_face(reference, reference_box, padding=face_crop_padding)
    reference_embedding = _embedding(
        reference_face,
        config.embedding_method,
        model_embedding_callback,
    )

    siamese_model: Any | None = None
    siamese_reference: np.ndarray | None = None
    siamese_inputs: list[np.ndarray] = []
    if config.liveness_method in {"siamese", "ensemble"}:
        siamese_model, siamese_model_status = _load_siamese_runtime(config)
        if siamese_model is not None:
            try:
                from .siamese_liveness import prepare_face

                siamese_reference = prepare_face(reference_face)
            except Exception:
                siamese_model = None
                siamese_model_status = {
                    "status": "unavailable",
                    "reason": "Не удалось подготовить эталонное лицо для Siamese-модели.",
                }
    else:
        siamese_model_status = {
            "status": "not_required",
            "reason": None,
        }

    suffix = f".{video_extension.lower().lstrip('.')}"
    with tempfile.TemporaryDirectory(prefix="ukz-verification-") as temp_dir:
        video_path = Path(temp_dir) / f"upload{suffix}"
        video_path.write_bytes(video_bytes)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise HTTPException(422, detail="Не удалось декодировать видео. Проверьте формат и кодек файла.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        declared_duration = declared_frames / fps if fps > 0 and declared_frames > 0 else 0.0
        observations: list[dict[str, Any]] = []
        green_samples: list[float] = []
        last_face_box: tuple[int, int, int, int] | None = None
        frame_index = 0
        sampled_frames = 0
        sharpness_values: list[float] = []
        brightness_values: list[float] = []
        face_area_values: list[float] = []
        multi_candidate_frames = 0
        skipped_multi_candidate_frames = 0
        pose_values: list[dict[str, float]] = []
        eye_states: list[bool] = []
        arcface_embedding_samples: list[np.ndarray] = []
        arcface_embedding_errors = 0
        arcface_candidate_crops: dict[int, np.ndarray] = {}
        sample_interval = max(1, min(
            config.frame_sample_interval,
            config.liveness_frame_sample_interval,
        ))
        if declared_frames <= 0:
            counted_frames = 0
            while capture.grab():
                counted_frames += 1
            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise HTTPException(
                    422,
                    detail="Не удалось повторно открыть видео для выборки кадров.",
                )
            declared_frames = counted_frames
            declared_duration = (
                declared_frames / fps if fps > 0 and declared_frames > 0 else 0.0
            )
        desired_samples = min(
            config.max_frames_to_process,
            max(1, math.ceil(declared_frames / sample_interval)),
        )
        sample_indices = set(
            _uniform_frame_indices(declared_frames, desired_samples)
        )
        effective_sample_interval = max(
            1,
            math.ceil(declared_frames / max(len(sample_indices), 1)),
        )
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index in sample_indices:
                    faces = _detect_faces(frame, config.detector_backend, robust=False)
                    sampled_frames += 1
                    if len(faces) > 1:
                        multi_candidate_frames += 1
                    if faces:
                        skip_multiple_candidate_frame = (
                            config.multi_face_strategy in {"skip", "sckip", "skip_frame"}
                            and len(faces) > config.max_faces_per_frame
                        )
                        if skip_multiple_candidate_frame:
                            selected_face = None
                            skipped_multi_candidate_frames += 1
                            last_face_box = None
                        else:
                            selected_face = _select_face_candidate(
                                frame,
                                faces,
                                reference_embedding,
                                config.embedding_method,
                                config.multi_face_strategy,
                                config.max_faces_per_frame,
                                face_crop_padding,
                                model_embedding_callback,
                            )
                        if selected_face is None:
                            eye_states.append(False)
                            last_face_box = None
                        else:
                            box, crop_color, _descriptor, raw_score = selected_face
                            last_face_box = box
                            x, y, face_width, face_height = box
                            crop = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
                            eyes = _eye_boxes(crop)
                            eye_states.append(len(eyes) / 2 >= (1 - config.blink_min_ratio))
                            pose = _estimate_pose(box, crop, eyes, frame.shape)
                            if pose:
                                pose_values.append(pose)
                            score = _cosine_score(raw_score)
                            texture = _texture_metrics(crop, config)
                            area_pct = float(face_width * face_height * 100 / max(1, width * height))
                            frame_quality = _evaluate_frame_quality(
                                crop,
                                area_pct,
                                pose,
                                config,
                            )
                            sharpness = frame_quality["sharpness"]
                            brightness = frame_quality["brightness"]
                            sharpness_values.append(sharpness)
                            brightness_values.append(brightness)
                            face_area_values.append(area_pct)
                            observations.append({
                                "frame_id": int(frame_index),
                                "face_box": (int(x), int(y), int(face_width), int(face_height)),
                                "score": score,
                                "raw_score": raw_score,
                                "texture": texture,
                                "sharpness": sharpness,
                                "brightness": brightness,
                                "area_pct": area_pct,
                                "pose": pose,
                                "eye_count": len(eyes),
                                "frame_quality": frame_quality,
                            })
                            if frame_quality["eligible"] and arcface_embedding_callback:
                                arcface_candidate_crops[frame_index] = crop_color.copy()
                            if siamese_model is not None:
                                from .siamese_liveness import prepare_face

                                siamese_inputs.append(prepare_face(crop_color))
                    else:
                        eye_states.append(False)
                        last_face_box = None

                if last_face_box is not None:
                    x, y, face_width, face_height = last_face_box
                    left = max(0, x + int(face_width * 0.25))
                    right = min(frame.shape[1], x + int(face_width * 0.75))
                    top = max(0, y + int(face_height * 0.25))
                    bottom = min(frame.shape[0], y + int(face_height * 0.62))
                    roi = frame[top:bottom, left:right]
                    if roi.size:
                        green_samples.append(float(np.mean(roi[:, :, 1])))
                frame_index += 1
        finally:
            capture.release()
            with suppress(FileNotFoundError):
                video_path.unlink()

        total_frames = frame_index or declared_frames
        duration = (
            frame_index / fps
            if fps > 0 and frame_index > 0
            else declared_duration
        )
        effective_sample_interval = max(
            1,
            math.ceil(total_frames / max(sampled_frames, 1)),
        )
        if sampled_frames == 0:
            raise HTTPException(422, detail="В видео не удалось прочитать ни одного кадра.")

    detected_frames = len(observations)
    siamese_scores: list[float] = []
    siamese_score: float | None = None
    if siamese_model is not None and siamese_reference is not None and siamese_inputs:
        try:
            from .siamese_liveness import score_face_pairs

            siamese_scores = score_face_pairs(
                siamese_model,
                siamese_reference,
                siamese_inputs,
            )
            for observation, score in zip(observations, siamese_scores):
                observation["siamese_score"] = round(score, 6)
            siamese_score = float(np.mean(siamese_scores))
        except Exception:
            siamese_model = None
            siamese_model_status = {
                "status": "unavailable",
                "reason": "Не удалось выполнить инференс Siamese-модели.",
            }

    selected_frame_observations = _select_aggregation_frames(observations, config)
    if arcface_embedding_callback:
        for observation in selected_frame_observations:
            crop_color = arcface_candidate_crops.get(observation["frame_id"])
            if crop_color is None:
                continue
            try:
                vector = np.asarray(
                    arcface_embedding_callback(crop_color),
                    dtype=np.float32,
                ).reshape(-1)
                if vector.size and np.isfinite(vector).all():
                    arcface_embedding_samples.append(vector)
            except Exception:
                arcface_embedding_errors += 1
    arcface_candidate_crops.clear()

    per_frame_scores = [round(float(item["score"]), 6) for item in observations]
    raw_cosine_scores = [float(item["raw_score"]) for item in observations]
    frame_ids = [item["frame_id"] for item in observations]
    selected_frame_scores = [
        round(float(item["score"]), 6) for item in selected_frame_observations
    ]
    selected_raw_cosine_scores = [
        float(item["raw_score"]) for item in selected_frame_observations
    ]
    match_score = (
        float(np.median(selected_frame_scores)) if selected_frame_scores else 0.0
    )
    raw_cosine_mean = float(np.mean(raw_cosine_scores)) if raw_cosine_scores else 0.0
    raw_cosine_median = (
        float(np.median(selected_raw_cosine_scores))
        if selected_raw_cosine_scores
        else 0.0
    )
    selected_frame_ids = [
        int(item["frame_id"]) for item in selected_frame_observations
    ]
    best_frame_id = (
        int(selected_frame_observations[
            int(np.argmax(selected_frame_scores))
        ]["frame_id"])
        if selected_frame_scores
        else None
    )
    required_usable_frames = max(config.min_detected_frames, config.frame_median_count)
    match_score_threshold = config.match_threshold  #_cosine_score(config.match_threshold)
    manual_review_score_threshold = config.manual_review_threshold #_cosine_score(config.manual_review_threshold)

    if len(selected_frame_observations) < required_usable_frames:
        match_decision = "insufficient_quality"
    elif match_score >= match_score_threshold:
        match_decision = "match"
    elif match_score >= manual_review_score_threshold:
        match_decision = "manual_review"
    else:
        match_decision = "no_match"

    blink_count = 0
    closed_run = 0
    has_opened = False
    for eyes_open in eye_states:
        if eyes_open:
            if has_opened and 1 <= closed_run <= 3:
                blink_count += 1
            has_opened = True
            closed_run = 0
        elif has_opened:
            closed_run += 1
            if closed_run > 3:
                has_opened = False
                closed_run = 0
    blink_score = float(min(1.0, blink_count / max(1, config.blink_min_count)))

    motion_frames = 0
    max_motion_delta = 0.0
    previous_box: tuple[int, int, int, int] | None = None
    for observation in observations:
        box = observation["face_box"]
        if previous_box:
            delta = float(sum(abs(current - previous) for current, previous in zip(box, previous_box)))
            max_motion_delta = max(max_motion_delta, delta)
            motion_floor = max(3.0, max(previous_box[2], previous_box[3]) * 0.02)
            if delta > motion_floor:
                motion_frames += 1
        previous_box = box
    motion_score = float(min(1.0, motion_frames / max(1, config.head_motion_min_frames)))

    texture_details: dict[str, Any] = {}
    if observations:
        texture_entropy = float(np.mean([item["texture"]["lbp_entropy"] for item in observations]))
        fft_peak_ratio = float(np.mean([item["texture"]["fft_peak_ratio"] for item in observations]))
        fft_suspicious = any(bool(item["texture"]["fft_suspicious"]) for item in observations)
        low_entropy = texture_entropy < config.texture_lbp_threshold
        texture_suspicious = low_entropy or fft_suspicious
        texture_score = float(np.clip(texture_entropy * (0.6 if fft_suspicious else 1.0), 0, 1))
        texture_details = {
            "lbp_entropy": round(texture_entropy, 4),
            "lbp_low_entropy": low_entropy,
            "fft_peak_ratio": round(fft_peak_ratio, 4),
            "fft_suspicious": fft_suspicious,
            "score": round(texture_score, 4),
            "method": "Higher LBP entropy supports natural texture; periodic FFT peaks reduce the score.",
        }
    else:
        texture_entropy = 0.0
        fft_peak_ratio = 0.0
        texture_suspicious = True
        texture_score = 0.0

    rppg = _estimate_rppg(green_samples, fps, config)
    liveness_score, liveness_decision, attack_type = _score_liveness(
        blink_score,
        motion_score,
        texture_score,
        rppg,
        config,
        siamese_score,
        siamese_model_status["status"] == "available",
    )
    configured_liveness_unavailable = attack_type == "configured_model_unavailable"
    liveness_component_unavailable = configured_liveness_unavailable
    liveness_unavailable_reason = (
        siamese_model_status["reason"]
        if config.liveness_method in {"siamese", "ensemble"}
        else "Компонент проверки живости недоступен."
    )
    if texture_suspicious:
        attack_type = "screen_or_print_texture_suspected"
    elif blink_count == 0 and motion_frames < config.head_motion_min_frames and attack_type is None:
        attack_type = "static_presentation_suspected"

    capture_report = capture_quality_report(width, height, fps, config)
    resolution_ok = capture_report["resolution_ok"]
    fps_ok = capture_report["fps_ok"]
    fps_acceptable = capture_report["fps_acceptable"]
    duration_ok = duration >= config.min_duration_sec
    face_detection_rate = detected_frames / max(1, sampled_frames)
    face_detection_ok = detected_frames >= config.min_detected_frames
    frame_selection_ok = len(selected_frame_observations) >= required_usable_frames
    frame_quality_rejection_counts: dict[str, int] = {}
    for observation in observations:
        for reason in observation["frame_quality"]["reasons"]:
            frame_quality_rejection_counts[reason] = (
                frame_quality_rejection_counts.get(reason, 0) + 1
            )
    mean_sharpness = float(np.mean(sharpness_values)) if sharpness_values else 0.0
    mean_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    mean_area_pct = float(np.mean(face_area_values)) if face_area_values else 0.0
    face_area_ok = mean_area_pct >= config.min_face_area_pct
    pose_yaw = max((abs(item["yaw"]) for item in pose_values), default=None)
    pose_pitch = max((abs(item["pitch"]) for item in pose_values), default=None)
    pose_roll = max((abs(item["roll"]) for item in pose_values), default=None)
    angles_measured = pose_yaw is not None
    angles_ok = bool(
        angles_measured
        and pose_yaw <= config.max_yaw
        and pose_pitch <= config.max_pitch
        and pose_roll <= config.max_roll
    )

    blocking_issues: list[str] = []
    warnings: list[str] = []
    if not resolution_ok:
        blocking_issues.append("Разрешение видео ниже минимального.")
    if not fps_acceptable:
        blocking_issues.append("Частота кадров ниже минимальной.")
    elif not fps_ok:
        warnings.append(
            "Частота кадров файла немного ниже минимума из-за допустимого округления метаданных."
        )
    if not duration_ok:
        blocking_issues.append("Длительность видео ниже минимальной.")
    if not face_detection_ok:
        blocking_issues.append(f"Лицо найдено менее чем в {config.min_detected_frames} кадрах.")
    if not frame_selection_ok:
        blocking_issues.append(
            "После фильтра качества осталось "
            f"{len(selected_frame_observations)} кадров; требуется "
            f"{required_usable_frames}."
        )
    if not face_area_ok:
        blocking_issues.append("Средняя площадь лица ниже установленного минимума.")
    if mean_sharpness < config.max_blur_variance:
        warnings.append("Видео может быть размытым.")
    if mean_brightness < 45 or mean_brightness > 220:
        warnings.append("Освещение вне рекомендуемого диапазона.")
    if angles_measured and not angles_ok:
        warnings.append("Обнаружен значительный поворот или наклон головы.")
    (
        multi_candidate_ratio,
        multi_candidate_level,
        multi_face_blocked,
    ) = _classify_multi_face_candidates(
        multi_candidate_frames,
        sampled_frames,
        config.multi_face_warn_threshold,
        config.multi_face_block_threshold,
    )
    if multi_candidate_ratio >= config.multi_face_warn_threshold and multi_candidate_frames > 0:
        warnings.append(
            "Доля кадров с несколькими лицами достигла порога предупреждения "
            f"({multi_candidate_ratio:.1%} при пороге {config.multi_face_warn_threshold:.1%})."
        )
    if multi_face_blocked:
        blocking_issues.append(
            "Доля кадров с несколькими лицами "
            f"({multi_candidate_ratio:.1%}) превышает порог блокировки "
            f"({config.multi_face_block_threshold:.1%})."
        )

    quality_components = [
        float(resolution_ok),
        float(fps_ok),
        float(duration_ok),
        float(face_detection_ok),
        min(1.0, len(selected_frame_observations) / max(required_usable_frames, 1)),
        min(1.0, mean_area_pct / max(config.min_face_area_pct, 1.0)),
        min(1.0, mean_sharpness / max(config.max_blur_variance * 3, 1.0)),
        1.0 if 45 <= mean_brightness <= 220 else 0.5,
        1.0 if angles_ok else 0.5,
    ]
    quality_score = float(np.clip(np.mean(quality_components), 0, 1))
    if quality_score < config.capture_quality_threshold:
        blocking_issues.append(
            "Общая оценка качества видео "
            f"({quality_score:.3f}) ниже порога "
            f"({config.capture_quality_threshold:.3f})."
        )
    quality_overall = "reject" if blocking_issues else "accept"
    quality_checks = {
        "resolution": capture_report["resolution"],
        "fps": capture_report["fps"],
        "duration": {
            "status": "pass" if duration_ok else "fail",
            "actual": round(duration, 2),
            "required": config.min_duration_sec,
        },
        "blur": {
            "status": "pass" if mean_sharpness >= config.max_blur_variance else "warning",
            "actual": round(mean_sharpness, 2),
            "required": config.max_blur_variance,
            "method": "variance of Laplacian; higher is sharper",
        },
        "frame_selection": {
            "status": "pass" if frame_selection_ok else "fail",
            "sampling_strategy": config.frame_sampling_strategy,
            "selection_strategy": config.frame_selection_strategy,
            "aggregation": "median",
            "requested_frames": config.frame_median_count,
            "required_usable_frames": required_usable_frames,
            "sampled_frames": sampled_frames,
            "detected_frames": detected_frames,
            "usable_frames": sum(
                bool(item["frame_quality"]["eligible"]) for item in observations
            ),
            "selected_frames": len(selected_frame_observations),
            "excluded_frames": sum(
                not bool(item["frame_quality"]["eligible"]) for item in observations
            ),
            "rejection_reasons": frame_quality_rejection_counts,
            "thresholds": {
                "minimum_sharpness": config.max_blur_variance,
                "minimum_contrast": config.frame_quality_min_contrast,
                "brightness_range": [
                    config.frame_quality_min_brightness,
                    config.frame_quality_max_brightness,
                ],
                "maximum_illumination_spread": (
                    config.frame_quality_max_illumination_spread
                ),
                "minimum_face_area_pct": config.min_face_area_pct,
            },
        },
        "face_visibility": {
            "status": "pass" if face_area_ok and face_detection_ok else "fail",
            "actual": round(mean_area_pct, 2),
            "required": config.min_face_area_pct,
            "detected_frames": detected_frames,
            "required_detected_frames": config.min_detected_frames,
            "detection_rate_pct": round(face_detection_rate * 100, 2),
        },
        "multi_face_candidates": {
            "status": "fail" if multi_face_blocked else "warning" if multi_candidate_ratio >= config.multi_face_warn_threshold and multi_candidate_frames > 0 else "pass",
            "actual": multi_candidate_frames,
            "ratio": round(multi_candidate_ratio, 4),
            "level": multi_candidate_level,
            "strategy": config.multi_face_strategy,
            "skipped_frames": skipped_multi_candidate_frames,
            "warn_threshold": config.multi_face_warn_threshold,
            "block_threshold": config.multi_face_block_threshold,
            "blocked": multi_face_blocked,
            "note": "Предупреждение показывается при любом обнаружении; блокировка только при превышении порога.",
        },
        "lighting": {
            "status": "pass" if 45 <= mean_brightness <= 220 else "warning",
            "actual": round(mean_brightness, 2),
            "note": "Средняя яркость ROI лица; lux не измеряется обычной веб-камерой.",
        },
        "angle": {
            "status": "pass" if angles_ok else "warning" if angles_measured else "not_assessed",
            "actual": None if pose_yaw is None else {
                "max_yaw": round(pose_yaw, 2),
                "max_pitch": round(float(pose_pitch), 2),
                "max_roll": round(float(pose_roll), 2),
            },
            "required": {
                "max_yaw": config.max_yaw,
                "max_pitch": config.max_pitch,
                "max_roll": config.max_roll,
            },
            "method": "approximate Haar eye geometry",
        },
        "occlusion": {
            "status": "not_assessed",
            "actual": None,
            "required": config.max_occlusion_pct,
            "note": "Каскад Haar не измеряет процент перекрытия лица надёжно.",
        },
    }

    status = _resolve_verification_status(
        liveness_component_unavailable=liveness_component_unavailable,
        blocking_issues=blocking_issues,
        match_decision=match_decision,
        match_score=match_score,
        liveness_decision=liveness_decision,
        quality_score=quality_score,
        quality_threshold=config.capture_quality_threshold,
        manual_review_threshold=config.manual_review_threshold,
    )
    not_evaluable_reason = liveness_unavailable_reason if status == "not_evaluable" else None
    decision = "match" if status == "verified" else "review" if status in {"manual_review", "insufficient_quality", "not_evaluable"} else "no_match"
    if status == "not_verified" and not attack_type:
        attack_type = "face_or_liveness_threshold_not_met"

    health_id = {
        "status": "not_calculated",
        "health_id_value": None,
        "completeness": 0.0,
        "uncertainty_score": 1.0,
        "category": "review",
        "evidence": {"rppg": rppg},
        "error": (
            "Этот endpoint рассчитывает биометрическую верификацию. "
            "Для HEALTH_ID нужны дополнительные измерения и отдельная исследовательская валидация."
        ),
    }
    config_hash = config.fingerprint()
    details = {
        "event_id": "",
        "model_version": model_version,
        "implementation_version": ALGORITHM_IMPLEMENTATION_VERSION,
        "config_hash": config_hash,
        "face_match": {
            "match_score": round(match_score, 6),
            "per_frame_scores": per_frame_scores,
            "frame_ids": frame_ids,
            "selected_frame_ids": selected_frame_ids,
            "selected_frame_scores": selected_frame_scores,
            "frame_quality": [
                {
                    "frame_id": int(item["frame_id"]),
                    "score": item["frame_quality"]["score"],
                    "eligible": item["frame_quality"]["eligible"],
                    "reasons": item["frame_quality"]["reasons"],
                    "sharpness": item["frame_quality"]["sharpness"],
                    "brightness": item["frame_quality"]["brightness"],
                    "contrast": item["frame_quality"]["contrast"],
                    "illumination_spread": item["frame_quality"]["illumination_spread"],
                }
                for item in observations
            ],
            "best_frame_id": best_frame_id,
            "aggregation": "median",
            "frame_sampling_strategy": config.frame_sampling_strategy,
            "frame_selection_strategy": config.frame_selection_strategy,
            "requested_frame_count": config.frame_median_count,
            "usable_frame_count": sum(
                bool(item["frame_quality"]["eligible"]) for item in observations
            ),
            "excluded_frame_count": sum(
                not bool(item["frame_quality"]["eligible"]) for item in observations
            ),
            "threshold": round(match_score_threshold, 6),
            "manual_review_threshold": round(manual_review_score_threshold, 6),
            "raw_cosine_threshold": config.match_threshold,
            "raw_cosine_manual_review_threshold": config.manual_review_threshold,
            "raw_cosine_mean": round(raw_cosine_mean, 6),
            "raw_cosine_median": round(raw_cosine_median, 6),
            "score_scale": "cosine mapped from [-1, 1] to [0, 1]",
            "decision": match_decision,
            "total_frames": int(total_frames),
            "detected_frames": int(detected_frames),
            "multi_face_strategy": config.multi_face_strategy,
            "multi_candidate_frames": int(multi_candidate_frames),
            "multi_candidate_ratio": round(multi_candidate_ratio, 4),
            "multi_candidate_level": multi_candidate_level,
            "multi_face_warn_threshold": config.multi_face_warn_threshold,
            "multi_face_block_threshold": config.multi_face_block_threshold,
            "multi_face_blocked": multi_face_blocked,
            "skipped_multi_candidate_frames": int(skipped_multi_candidate_frames),
            "sampled_frames": int(sampled_frames),
            "effective_sample_interval_frames": int(effective_sample_interval),
            "reference_embedding_norm": round(float(np.linalg.norm(reference_embedding)), 6),
            "detector_backend": config.detector_backend,
            "embedding_method": config.embedding_method,
        },
        "liveness": {
            "liveness_score": liveness_score,
            "decision": liveness_decision,
            "method": config.liveness_method,
            "strategy": config.liveness_strategy,
            "aggregation_weights": {
                "siamese": config.liveness_siamese_weight,
                "blink": config.blink_weight,
                "head_motion": config.head_motion_weight,
                "texture": config.texture_weight,
                "depth": config.depth_weight,
                "rppg_if_available": config.liveness_rppg_weight,
            },
            "model_status": {
                "siamese": siamese_model_status,
                "rppg": rppg["status"],
            },
            "liveness_signals": {
                "siamese": {
                    "score": None if siamese_score is None else round(siamese_score, 4),
                    "passed": (
                        None
                        if siamese_score is None
                        else siamese_score >= config.liveness_siamese_threshold
                    ),
                    "details": {
                        "frames_scored": len(siamese_scores),
                        "threshold": config.liveness_siamese_threshold,
                        "weight": config.liveness_siamese_weight,
                        "status": siamese_model_status["status"],
                    },
                },
                "blink": {
                    "score": round(blink_score, 4),
                    "passed": blink_score >= config.liveness_threshold,
                    "details": {
                        "blink_count": blink_count,
                        "minimum_required": config.blink_min_count,
                        "eye_states_sampled": len(eye_states),
                        "method": "Haar eye detections across sampled frames",
                    },
                },
                "head_motion": {
                    "score": round(motion_score, 4),
                    "passed": motion_score >= config.liveness_threshold,
                    "details": {
                        "motion_frames": motion_frames,
                        "total_frames": len(pose_values),
                        "max_observed_delta": round(max_motion_delta, 2),
                    },
                },
                "texture": {
                    "score": round(texture_score, 4),
                    "passed": not texture_suspicious,
                    "details": texture_details,
                },
                "depth": {
                    "score": 0.5,
                    "passed": not config.depth_enabled,
                    "details": {"reason": "disabled" if not config.depth_enabled else "no depth sensor in uploaded video"},
                },
                "rppg": {
                    "score": round(float(rppg["confidence"]), 4),
                    "passed": rppg["status"] == "available",
                    "details": rppg,
                },
            },
            "evidence_frames": [
                item["frame_id"]
                for item in observations
                if item["texture"]["fft_suspicious"]
            ][:20],
            "attack_type": attack_type,
        },
        "quality": {
            "overall": quality_overall,
            "checks": quality_checks,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
        },
        "route": status,
        "health_id": health_id,
        "disclaimer": DISCLAIMER,
    }
    if status == "not_evaluable":
        details["not_evaluable_reason"] = not_evaluable_reason
        details["recommended_action"] = "retry_or_manual_review"
        details["face_match"]["decision"] = "not_evaluable"
        details["liveness"]["liveness_score"] = None
        details["liveness"]["decision"] = "not_evaluable"
    elif status == "insufficient_quality":
        details["recommended_action"] = "retry_or_manual_review"
    return {
        "status": status,
        "match_score": round(match_score, 6),
        "liveness_score": None if status == "not_evaluable" else liveness_score,
        "quality_score": round(quality_score, 4),
        "decision": decision,
        "attack_type": None if status == "not_evaluable" else attack_type,
        "details": details,
        "arcface_embedding_samples": arcface_embedding_samples,
        "arcface_embedding_errors": arcface_embedding_errors,
    }