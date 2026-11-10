from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    # Face match
    match_threshold: float = 0.85
    manual_review_threshold: float = 0.60
    max_frames_to_process: int = 120
    frame_sample_interval: int = 5
    liveness_frame_sample_interval: int = 2
    frame_sampling_strategy: str = "uniform_full_video"
    frame_selection_strategy: str = "best_quality_median"
    frame_median_count: int = 5
    frame_quality_min_contrast: float = 12.0
    frame_quality_min_brightness: float = 45.0
    frame_quality_max_brightness: float = 220.0
    frame_quality_max_illumination_spread: float = 90.0
    min_detected_frames: int = 5
    embedding_method: str = "hog_lbp" # Варианты: hog_lbp, arcface, resnet50_arcface
    detector_backend: str = "haar"    # Варианты: "haar", "mtcnn", "retinaface"
    # мультифейс проверка (верификация 1:1)
    max_faces_per_frame: int = 1
    multi_face_strategy: str = "select_closest_to_reference"
    multi_face_warn_threshold: float = 0.30
    multi_face_block_threshold: float = 0.50

    # Liveness
    liveness_pass: float = 0.70
    liveness_fail: float = 0.50
    blink_weight: float = 0.35
    head_motion_weight: float = 0.30
    texture_weight: float = 0.25
    depth_weight: float = 0.10
    blink_min_count: int = 1
    blink_min_ratio: float = 0.25
    head_motion_min_frames: int = 3
    texture_lbp_threshold: float = 0.45
    texture_freq_threshold: float = 0.40
    depth_enabled: bool = False
    liveness_method: str = "ensemble"
    liveness_strategy: str = "adaptive"
    liveness_threshold: float = 0.50
    liveness_siamese_model_path: str = "models/siamese_liveness.pt"
    liveness_siamese_threshold: float = 0.50
    liveness_siamese_weight: float = 0.40
    liveness_rppg_min_confidence: float = 0.30
    liveness_rppg_weight: float = 0.60

    # Capture quality; min_resolution is (width, height).
    min_resolution: tuple[int, int] = (640, 480)
    min_fps: float = 15.0
    min_duration_sec: float = 3.0
    min_face_area_pct: float = 5.0
    max_blur_variance: float = 35.0
    max_yaw: float = 25.0
    max_pitch: float = 20.0
    max_roll: float = 15.0
    max_occlusion_pct: float = 10.0
    capture_quality_threshold: float = 0.85

    # rPPG is optional; these thresholds come from the project specification.
    rppg_min_duration_sec: float = 10.0
    rppg_target_fps: float = 30.0
    rppg_heart_rate_band: tuple[float, float] = (0.70, 3.0)
    rppg_breathing_band: tuple[float, float] = (0.15, 0.50)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def public_settings(self) -> dict[str, Any]:
        return asdict(self)


MODEL_CONFIG = ModelConfig()

PARAMETER_OPTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "liveness_method": (
        ("siamese", "siamese"),
        ("rppg", "rppg"),
        ("ensemble", "ensemble"),
    ),
    "liveness_strategy": (
        ("weighted_avg", "weighted_avg"),
        ("and", "and"),
        ("or", "or"),
        ("adaptive", "adaptive"),
    ),
    "embedding_method": (
        ("hog_lbp", "HOG + LBP — по умолчанию"),
        ("arcface", "ArcFace — нейросетевые эмбеддинги"),
        ("resnet50_arcface", "ResNet50 + ArcFace — обученная модель проекта"),
    ),
    "detector_backend": (
        ("haar", "Каскад Хаара — по умолчанию"),
        ("mtcnn", "MTCNN"),
        ("retinaface", "RetinaFace"),
    ),
    "frame_sampling_strategy": (
        ("uniform_full_video", "Равномерно по всей записи"),
    ),
    "frame_selection_strategy": (
        ("best_quality_median", "Лучшие кадры по качеству, медиана"),
        ("all_accepted_median", "Медиана всех прошедших фильтр кадров"),
    ),
    "multi_face_strategy": (
        ("max", "max"),
        ("skip_frame", "skip_frame"),
        ("select_closest_to_reference", "select_closest_to_reference"),
    ),
}

PARAMETER_OPTION_VALUES = {
    key: frozenset(value for value, _label in options)
    for key, options in PARAMETER_OPTIONS.items()
}


_PARAMETER_LABELS = {
    "match_threshold": "Порог исходного cosine для совпадения",
    "manual_review_threshold": "Порог исходного cosine для ручной проверки",
    "max_frames_to_process": "Максимум кадров для обработки",
    "frame_sample_interval": "Интервал выборки кадров",
    "liveness_frame_sample_interval": "Интервал кадров для проверки живости",
    "frame_sampling_strategy": "Стратегия выборки по длительности видео",
    "frame_selection_strategy": "Стратегия отбора кадров для сопоставления",
    "frame_median_count": "Число лучших кадров для медианы (5 или 10)",
    "frame_quality_min_contrast": "Минимальный контраст лица",
    "frame_quality_min_brightness": "Минимальная яркость лица",
    "frame_quality_max_brightness": "Максимальная яркость лица",
    "frame_quality_max_illumination_spread": "Допустимая неравномерность освещения",
    "min_detected_frames": "Минимум кадров с обнаруженным лицом",
    "embedding_method": "Метод представления лица",
    "detector_backend": "Детектор лица",
    "liveness_pass": "Порог подтверждения живого присутствия",
    "liveness_fail": "Порог отклонения проверки живого присутствия",
    "blink_weight": "Вес сигнала моргания",
    "head_motion_weight": "Вес движения головы",
    "texture_weight": "Вес анализа текстуры",
    "depth_weight": "Вес оценки глубины",
    "blink_min_count": "Минимум морганий",
    "blink_min_ratio": "Минимальная доля моргания",
    "head_motion_min_frames": "Минимум кадров с движением головы",
    "texture_lbp_threshold": "Порог текстурного анализа LBP",
    "texture_freq_threshold": "Порог частотного анализа текстуры",
    "depth_enabled": "Включить оценку глубины",
    "liveness_method": "Метод проверки живого присутствия",
    "liveness_strategy": "Стратегия объединения сигналов",
    "liveness_threshold": "Общий порог живого присутствия",
    "liveness_siamese_model_path": "Путь к Siamese-модели",
    "liveness_siamese_threshold": "Порог Siamese-модели",
    "liveness_siamese_weight": "Вес Siamese-модели",
    "liveness_rppg_min_confidence": "Минимальная уверенность rPPG",
    "liveness_rppg_weight": "Вес сигнала rPPG",
    "min_resolution": "Минимальное разрешение (ширина, высота)",
    "min_fps": "Минимальная частота кадров",
    "min_duration_sec": "Минимальная длительность видео, сек",
    "min_face_area_pct": "Минимальная площадь лица, %",
    "max_blur_variance": "Минимальная резкость кадра",
    "max_yaw": "Максимальный поворот головы по yaw, °",
    "max_pitch": "Максимальный наклон головы по pitch, °",
    "max_roll": "Максимальный наклон головы по roll, °",
    "max_occlusion_pct": "Допустимое перекрытие лица, %",
    "rppg_min_duration_sec": "Минимальная длительность сигнала rPPG, сек",
    "rppg_target_fps": "Целевая частота кадров rPPG",
    "rppg_heart_rate_band": "Частотный диапазон пульса rPPG, Гц",
    "rppg_breathing_band": "Частотный диапазон дыхания rPPG, Гц",
    "max_faces_per_frame": "Максимум кандидатов до пропуска кадра",
    "multi_face_strategy": "Стратегия выбора лица при нескольких кандидатах",
    "multi_face_warn_threshold": "Порог доли кадров с несколькими лицами для предупреждения",
    "multi_face_block_threshold": "Порог доли кадров с несколькими лицами для блокировки проверки",
    "capture_quality_threshold": "Порог оценки качества видео",
}


def parameter_schema() -> list[dict[str, Any]]:
    """Return editable parameter metadata directly from the system config defaults."""
    groups = {
        "face": "Сопоставление лица",
        "liveness": "Проверка живого присутствия",
        "quality": "Качество записи",
        "rppg": "Анализ rPPG",
    }
    face_keys = {
        "match_threshold", "manual_review_threshold", "max_frames_to_process",
        "frame_sample_interval", "frame_sampling_strategy", "frame_selection_strategy",
        "frame_median_count", "min_detected_frames", "max_faces_per_frame",
        "multi_face_strategy", "multi_face_warn_threshold",
        "multi_face_block_threshold", "embedding_method", "detector_backend",
    }
    quality_keys = {
        "min_resolution", "min_fps", "min_duration_sec", "min_face_area_pct",
        "max_blur_variance", "max_yaw", "max_pitch", "max_roll", "max_occlusion_pct",
        "frame_quality_min_contrast", "frame_quality_min_brightness",
        "frame_quality_max_brightness", "frame_quality_max_illumination_spread",
    }
    fields = []
    for name, value in MODEL_CONFIG.public_settings().items():
        group = (
            "face" if name in face_keys
            else "quality" if name in quality_keys
            else "rppg" if name.startswith("rppg_") or name.startswith("liveness_rppg_")
            else "liveness"
        )
        options = PARAMETER_OPTIONS.get(name)
        value_type = (
            "boolean" if isinstance(value, bool)
            else "number" if isinstance(value, (int, float))
            else "array" if isinstance(value, tuple)
            else "select" if options
            else "text"
        )
        field = {
            "key": name,
            "label": _PARAMETER_LABELS.get(name, name),
            "group": groups[group],
            "value_type": value_type,
            "default": list(value) if isinstance(value, tuple) else value,
        }
        if options:
            field["options"] = [
                {"value": option_value, "label": label}
                for option_value, label in options
            ]
        fields.append(field)
    return fields