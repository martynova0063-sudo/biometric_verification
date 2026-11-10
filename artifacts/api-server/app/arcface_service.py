from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .arcface_model import ResNet50ArcFace


class ArcFaceIndexError(ValueError):
    """Raised when the persisted employee vectors cannot form a valid index."""


class ArcFaceImageError(ValueError):
    """Raised when an uploaded image cannot be used to create an embedding."""


def _index_synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._index_lock:
            return method(self, *args, **kwargs)
    return wrapped


def _atomic_save_array_pair(
    first_path: Path,
    first_array: np.ndarray,
    second_path: Path,
    second_array: np.ndarray,
) -> None:
    first_path.parent.mkdir(parents=True, exist_ok=True)
    second_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, value in (
            (first_path, first_array),
            (second_path, second_array),
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npy",
                prefix=f".{destination.stem}.",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                np.save(temporary, value, allow_pickle=False)
                temporary.flush()
                os.fsync(temporary.fileno())
                staged.append((Path(temporary.name), destination))
        for temporary_path, destination in staged:
            os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


class ArcFaceSearchService:
    max_image_bytes = 10 * 1024 * 1024

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index: faiss.IndexFlatL2 | None = None
        self.embeddings: np.ndarray | None = None
        self.uuids: np.ndarray | None = None
        self.enrolled_embeddings: np.ndarray | None = None
        self.enrolled_uuids: np.ndarray | None = None
        self.video_embeddings: np.ndarray | None = None
        self.video_uuids: np.ndarray | None = None
        self.model: ResNet50ArcFace | None = None
        self.index_error: str | None = None
        self.model_error: str | None = None
        self._index_lock = threading.RLock()
        self._model_lock = threading.Lock()
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @_index_synchronized
    def load_index(self) -> dict[str, int]:
        embeddings_path = self.storage_dir / "employee_embeddings.npy"
        uuids_path = self.storage_dir / "employee_uuids.npy"
        if not embeddings_path.is_file() or not uuids_path.is_file():
            self.index_error = (
                "Файлы employee_embeddings.npy и employee_uuids.npy "
                f"должны находиться в {self.storage_dir}."
            )
            raise FileNotFoundError(self.index_error)

        try:
            embeddings = np.load(embeddings_path, allow_pickle=False).astype(
                np.float32, copy=False
            )
            uuids = np.load(uuids_path, allow_pickle=False).astype(str)
        except (OSError, ValueError) as error:
            self.index_error = f"Не удалось прочитать файлы индекса: {error}"
            raise ArcFaceIndexError(self.index_error) from error

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            self.index_error = "Массив эмбеддингов должен быть непустой матрицей."
            raise ArcFaceIndexError(self.index_error)
        if uuids.ndim != 1 or len(uuids) != embeddings.shape[0]:
            self.index_error = (
                "Число UUID сотрудников не совпадает с числом эмбеддингов."
            )
            raise ArcFaceIndexError(self.index_error)
        if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
            self.index_error = "Эмбеддинги содержат неверную размерность или значения."
            raise ArcFaceIndexError(self.index_error)
        if any(not value for value in uuids):
            self.index_error = "Массив UUID содержит пустые значения."
            raise ArcFaceIndexError(self.index_error)

        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 1e-12):
            self.index_error = "Нельзя индексировать нулевой эмбеддинг."
            raise ArcFaceIndexError(self.index_error)
        embeddings = np.ascontiguousarray(embeddings / norms[:, None], dtype=np.float32)

        enrolled_embeddings_path = self.storage_dir / "employee_enrolled_embeddings.npy"
        enrolled_uuids_path = self.storage_dir / "employee_enrolled_uuids.npy"
        has_enrolled_embeddings = enrolled_embeddings_path.is_file()
        has_enrolled_uuids = enrolled_uuids_path.is_file()
        if has_enrolled_embeddings != has_enrolled_uuids:
            self.index_error = "Файлы дополнительных эмбеддингов ArcFace неполные."
            raise ArcFaceIndexError(self.index_error)

        if has_enrolled_embeddings:
            try:
                enrolled_embeddings = np.load(
                    enrolled_embeddings_path, allow_pickle=False
                ).astype(np.float32, copy=False)
                enrolled_uuids = np.load(
                    enrolled_uuids_path, allow_pickle=False
                ).astype(str)
            except (OSError, ValueError) as error:
                self.index_error = f"Не удалось прочитать дополнительные эмбеддинги: {error}"
                raise ArcFaceIndexError(self.index_error) from error

            if (
                enrolled_embeddings.ndim != 2
                or enrolled_embeddings.shape[0] == 0
                or enrolled_embeddings.shape[1] != embeddings.shape[1]
                or enrolled_uuids.ndim != 1
                or len(enrolled_uuids) != enrolled_embeddings.shape[0]
                or not np.isfinite(enrolled_embeddings).all()
                or any(not value for value in enrolled_uuids)
            ):
                self.index_error = "Дополнительные эмбеддинги имеют неверный формат."
                raise ArcFaceIndexError(self.index_error)
            enrolled_norms = np.linalg.norm(enrolled_embeddings, axis=1)
            if np.any(enrolled_norms <= 1e-12):
                self.index_error = "Нельзя индексировать нулевой дополнительный эмбеддинг."
                raise ArcFaceIndexError(self.index_error)
            enrolled_embeddings = np.ascontiguousarray(
                enrolled_embeddings / enrolled_norms[:, None],
                dtype=np.float32,
            )
            embeddings = np.ascontiguousarray(
                np.concatenate((embeddings, enrolled_embeddings), axis=0),
                dtype=np.float32,
            )
            uuids = np.concatenate((uuids, enrolled_uuids))
        else:
            enrolled_embeddings = np.empty((0, embeddings.shape[1]), dtype=np.float32)
            enrolled_uuids = np.empty((0,), dtype=str)

        video_embeddings_path = self.storage_dir / "employee_video_embeddings.npy"
        video_uuids_path = self.storage_dir / "employee_video_uuids.npy"
        has_video_embeddings = video_embeddings_path.is_file()
        has_video_uuids = video_uuids_path.is_file()
        if has_video_embeddings != has_video_uuids:
            self.index_error = "Файлы видеоэмбеддингов ArcFace неполные."
            raise ArcFaceIndexError(self.index_error)

        if has_video_embeddings:
            try:
                video_embeddings = np.load(
                    video_embeddings_path, allow_pickle=False
                ).astype(np.float32, copy=False)
                video_uuids = np.load(
                    video_uuids_path, allow_pickle=False
                ).astype(str)
            except (OSError, ValueError) as error:
                self.index_error = f"Не удалось прочитать видеоэмбеддинги: {error}"
                raise ArcFaceIndexError(self.index_error) from error

            if (
                video_embeddings.ndim != 2
                or video_embeddings.shape[0] == 0
                or video_embeddings.shape[1] != embeddings.shape[1]
                or video_uuids.ndim != 1
                or len(video_uuids) != video_embeddings.shape[0]
                or not np.isfinite(video_embeddings).all()
                or any(not value for value in video_uuids)
            ):
                self.index_error = "Видеоэмбеддинги имеют неверный формат."
                raise ArcFaceIndexError(self.index_error)
            video_norms = np.linalg.norm(video_embeddings, axis=1)
            if np.any(video_norms <= 1e-12):
                self.index_error = "Нельзя индексировать нулевой видеоэмбеддинг."
                raise ArcFaceIndexError(self.index_error)
            video_embeddings = np.ascontiguousarray(
                video_embeddings / video_norms[:, None],
                dtype=np.float32,
            )
            embeddings = np.ascontiguousarray(
                np.concatenate((embeddings, video_embeddings), axis=0),
                dtype=np.float32,
            )
            uuids = np.concatenate((uuids, video_uuids))
        else:
            video_embeddings = np.empty((0, embeddings.shape[1]), dtype=np.float32)
            video_uuids = np.empty((0,), dtype=str)

        if self.model is not None:
            model_dim = self.model.embedding[0].out_features
            if embeddings.shape[1] != model_dim:
                self.index_error = (
                    f"Размерность индекса ({embeddings.shape[1]}) не совпадает "
                    f"с размерностью модели ({model_dim})."
                )
                raise ArcFaceIndexError(self.index_error)

        candidate_index = faiss.IndexFlatL2(int(embeddings.shape[1]))
        candidate_index.add(embeddings)

        self.embeddings = embeddings
        self.uuids = uuids
        self.enrolled_embeddings = enrolled_embeddings
        self.enrolled_uuids = enrolled_uuids
        self.video_embeddings = video_embeddings
        self.video_uuids = video_uuids
        self.index = candidate_index
        self.index_error = None
        return {
            "vectors_count": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
            "employees_count": int(len(np.unique(uuids))),
        }

    def load_model(self) -> None:
        model_path = self.storage_dir / "best_arcface_resnet50.pth"
        if not model_path.is_file():
            self.model_error = f"Файл модели не найден: {model_path.name}."
            raise FileNotFoundError(self.model_error)

        try:
            checkpoint: dict[str, Any] = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            num_classes = int(checkpoint["num_classes"])
            state_dict = checkpoint["model_state_dict"]
            if num_classes < 1 or not isinstance(state_dict, dict):
                raise ValueError("В файле модели отсутствуют веса или классы.")
            model = ResNet50ArcFace(
                num_classes=num_classes,
                pretrained=False,
            ).to(self.device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            self.model_error = f"Не удалось загрузить модель ArcFace: {error}"
            raise ArcFaceIndexError(self.model_error) from error

        model_dim = model.embedding[0].out_features
        if self.index is not None and self.index.d != model_dim:
            self.model_error = (
                f"Размерность модели ({model_dim}) не совпадает "
                f"с размерностью индекса ({self.index.d})."
            )
            raise ArcFaceIndexError(self.model_error)
        self.model = model
        self.model_error = None

    @_index_synchronized
    def status(self) -> dict[str, Any]:
        return {
            "index_ready": self.index is not None,
            "model_ready": self.model is not None,
            "vectors_count": int(self.index.ntotal) if self.index is not None else 0,
            "video_vectors_count": (
                int(len(self.video_embeddings))
                if self.video_embeddings is not None
                else 0
            ),
            "dim": int(self.index.d) if self.index is not None else None,
            "employees_count": (
                int(len(np.unique(self.uuids))) if self.uuids is not None else 0
            ),
            "class_mapping_ready": (
                self.storage_dir / "class_mapping.json"
            ).is_file(),
            "index_error": self.index_error,
            "model_error": self.model_error,
        }

    @_index_synchronized
    def search_embedding(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        if self.index is None or self.uuids is None:
            raise RuntimeError("Индекс сотрудников пока не загружен.")

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.index.d or not np.isfinite(query).all():
            raise ArcFaceIndexError("Размерность эмбеддинга не совпадает с индексом.")
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            raise ArcFaceIndexError("Нельзя выполнить поиск по нулевому эмбеддингу.")
        query = np.ascontiguousarray(query / norm, dtype=np.float32)

        # IndexFlatL2 returns squared L2 distances. For normalized vectors:
        # cosine similarity = 1 - squared_distance / 2.
        distances, positions = self.index.search(query, int(self.index.ntotal))
        best_by_uuid: dict[str, float] = {}
        for distance, position in zip(distances[0], positions[0]):
            if position < 0:
                continue
            employee_uuid = str(self.uuids[position])
            similarity = float(np.clip(1.0 - float(distance) / 2.0, -1.0, 1.0))
            best_by_uuid[employee_uuid] = max(
                similarity, best_by_uuid.get(employee_uuid, -1.0)
            )

        results = [
            {"uuid": employee_uuid, "cosine_similarity": similarity}
            for employee_uuid, similarity in sorted(
                best_by_uuid.items(), key=lambda item: item[1], reverse=True
            )[:top_k]
        ]
        top_match = results[0] if results else None
        return {
            "found": bool(
                top_match and top_match["cosine_similarity"] >= threshold
            ),
            "threshold": float(threshold),
            "top_match": top_match,
            "results": results,
        }

    @_index_synchronized
    def compare_video_embeddings(
        self,
        employee_uuid: str,
        video_embeddings: list[np.ndarray],
    ) -> dict[str, Any]:
        """Compare transient video vectors with this employee's reference vectors."""
        base = {
            "method": "ArcFace ResNet50",
            "reference_vectors": 0,
            "video_vectors": 0,
            "dimensions": self.index.d if self.index is not None else 512,
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
        }
        if self.model is None or self.index is None or self.embeddings is None or self.uuids is None:
            return {
                **base,
                "status": "unavailable",
                "message": "Модель или индекс ArcFace недоступны.",
            }

        reference = self.embeddings[self.uuids == employee_uuid]
        base["reference_vectors"] = int(len(reference))
        if not len(reference):
            return {
                **base,
                "status": "no_reference",
                "message": "Для сотрудника нет эталонных векторов в индексе.",
            }

        valid_video_vectors = []
        for vector in video_embeddings:
            candidate = np.asarray(vector, dtype=np.float32).reshape(-1)
            if candidate.size != self.index.d or not np.isfinite(candidate).all():
                continue
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-12:
                valid_video_vectors.append(candidate / norm)
        if not valid_video_vectors:
            return {
                **base,
                "status": "no_video_vectors",
                "message": "Из кадров с найденным лицом не удалось получить векторы ArcFace.",
            }

        current = np.ascontiguousarray(np.stack(valid_video_vectors), dtype=np.float32)
        base["video_vectors"] = int(len(current))
        reference = np.ascontiguousarray(reference, dtype=np.float32)
        reference_mean = reference.mean(axis=0)
        video_mean = current.mean(axis=0)
        reference_component_mean = float(reference.mean())
        video_component_mean = float(current.mean())
        reference_mean_norm = float(np.linalg.norm(reference_mean))
        video_mean_norm = float(np.linalg.norm(video_mean))
        centroid_similarity = (
            float(np.clip(np.dot(reference_mean, video_mean) / (reference_mean_norm * video_mean_norm), -1.0, 1.0))
            if reference_mean_norm > 1e-12 and video_mean_norm > 1e-12
            else None
        )

        mean_shift = reference_mean - video_mean
        reference_variance = (
            float(np.var(reference, ddof=1)) if len(reference) > 1 else None
        )
        video_variance = float(np.var(current, ddof=1)) if len(current) > 1 else None
        variance_ratio = (
            video_variance / reference_variance
            if video_variance is not None
            and reference_variance is not None
            and reference_variance > 1e-12
            else None
        )
        base["mean"] = {
            "reference": round(reference_component_mean, 8),
            "video": round(video_component_mean, 8),
            "delta": round(video_component_mean - reference_component_mean, 8),
            "centroid_cosine_similarity": (
                round(centroid_similarity, 6) if centroid_similarity is not None else None
            ),
        }
        base["variance"] = {
            "reference": round(reference_variance, 10) if reference_variance is not None else None,
            "video": round(video_variance, 10) if video_variance is not None else None,
            "ratio": round(variance_ratio, 4) if variance_ratio is not None else None,
        }

        if reference_mean_norm > 1e-12:
            reference_center = reference_mean / reference_mean_norm
            reference_distances = np.clip(1.0 - reference @ reference_center, 0.0, 2.0)
            video_distances = np.clip(1.0 - current @ reference_center, 0.0, 2.0)
            if len(reference) >= 5:
                median_distance = float(np.median(reference_distances))
                mad = float(np.median(np.abs(reference_distances - median_distance)))
                outlier_limit = median_distance + 3.0 * 1.4826 * mad
                outlier_limit = max(
                    outlier_limit,
                    float(np.quantile(reference_distances, 0.95)),
                )
                base["outliers"] = {
                    "count": int(np.sum(video_distances > outlier_limit + 1e-8)),
                    "total": int(len(current)),
                    "threshold": round(outlier_limit, 6),
                }

        pooled = np.concatenate((reference.reshape(-1), current.reshape(-1)))
        bound = max(float(np.max(np.abs(pooled))), 0.02)
        edges = np.linspace(-bound, bound, 17)
        base["distribution"] = {
            "bin_centers": np.round((edges[:-1] + edges[1:]) / 2, 6).tolist(),
            "reference_counts": np.histogram(reference, bins=edges)[0].astype(int).tolist(),
            "video_counts": np.histogram(current, bins=edges)[0].astype(int).tolist(),
        }

        enough_reference = len(reference) >= 2
        message = (
            "Сравнение исследовательское; статистика выбросов доступна при наличии не менее пяти эталонных векторов."
            if len(reference) >= 5
            else "Мало эталонных векторов: надёжно оценить дисперсию и выбросы нельзя."
        )
        return {
            **base,
            "status": "available" if enough_reference else "insufficient_reference",
            "message": message,
        }

    def embedding_from_image(self, image_bytes: bytes) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель ArcFace пока не загружена.")
        if not image_bytes or len(image_bytes) > self.max_image_bytes:
            raise ArcFaceImageError("Размер изображения должен быть от 1 байта до 10 МБ.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                if source.width * source.height > 25_000_000:
                    raise ArcFaceImageError(
                        "Размер изображения не должен превышать 25 мегапикселей."
                    )
                image = source.convert("RGB")
        except ArcFaceImageError:
            raise
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
            raise ArcFaceImageError("Не удалось прочитать изображение.") from error

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._model_lock, torch.inference_mode():
            embedding = self.model(input_tensor).cpu().numpy().astype(np.float32)
        embedding = embedding.reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(embedding).all() or norm <= 1e-12:
            raise ArcFaceIndexError("Модель вернула нулевой или некорректный эмбеддинг.")
        return np.ascontiguousarray(embedding / norm, dtype=np.float32)

    def embedding_from_face_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        """Create an ArcFace vector from a detected BGR face crop."""
        if self.model is None:
            raise RuntimeError("Модель ArcFace пока не загружена.")
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3 or face_bgr.size == 0:
            raise ArcFaceImageError("Не удалось прочитать кадр с лицом.")
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._model_lock, torch.inference_mode():
            embedding = self.model(input_tensor).cpu().numpy().astype(np.float32)
        embedding = embedding.reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(embedding).all() or norm <= 1e-12:
            raise ArcFaceIndexError("Модель вернула нулевой или некорректный эмбеддинг.")
        return np.ascontiguousarray(embedding / norm, dtype=np.float32)

    @_index_synchronized
    def add_embedding(
        self,
        employee_uuid: str,
        embedding: np.ndarray,
    ) -> dict[str, Any]:
        if not employee_uuid:
            raise ArcFaceIndexError("UUID сотрудника не должен быть пустым.")
        if self.index is None or self.embeddings is None or self.uuids is None:
            raise RuntimeError("Индекс сотрудников пока не загружен.")
        if (
            self.enrolled_embeddings is None
            or self.enrolled_uuids is None
            or self.video_embeddings is None
            or self.video_uuids is None
        ):
            raise RuntimeError("Состояние дополнительных эмбеддингов не загружено.")

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.index.d or not np.isfinite(vector).all():
            raise ArcFaceIndexError("Размерность эмбеддинга не совпадает с индексом.")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ArcFaceIndexError("Нельзя добавить нулевой эмбеддинг.")
        vector = np.ascontiguousarray(vector / norm, dtype=np.float32)

        same_employee = self.uuids == employee_uuid
        if np.any(same_employee):
            similarities = self.embeddings[same_employee] @ vector
            if float(np.max(similarities)) >= 0.99999:
                return {
                    "added": False,
                    "updated": False,
                    "already_indexed": True,
                    "vectors_count": int(self.index.ntotal),
                    "employees_count": int(len(np.unique(self.uuids))),
                }

        enrolled_positions = np.flatnonzero(self.enrolled_uuids == employee_uuid)
        updated = len(enrolled_positions) > 0
        base_count = (
            len(self.embeddings)
            - len(self.enrolled_embeddings)
            - len(self.video_embeddings)
        )
        if updated:
            enrolled_position = int(enrolled_positions[0])
            new_enrolled_embeddings = self.enrolled_embeddings.copy()
            new_enrolled_embeddings[enrolled_position] = vector
            new_enrolled_uuids = self.enrolled_uuids.copy()
        else:
            new_enrolled_embeddings = np.concatenate(
                (self.enrolled_embeddings, vector.reshape(1, -1)),
                axis=0,
            )
            new_enrolled_uuids = np.concatenate(
                (self.enrolled_uuids, np.asarray([employee_uuid], dtype=str))
            )

        new_embeddings = np.ascontiguousarray(
            np.concatenate(
                (self.embeddings[:base_count], new_enrolled_embeddings, self.video_embeddings),
                axis=0,
            ),
            dtype=np.float32,
        )
        new_uuids = np.concatenate(
            (
                self.uuids[:base_count],
                new_enrolled_uuids,
                self.video_uuids,
            )
        )

        candidate_index = faiss.IndexFlatL2(int(new_embeddings.shape[1]))
        candidate_index.add(np.ascontiguousarray(new_embeddings, dtype=np.float32))
        _atomic_save_array_pair(
            self.storage_dir / "employee_enrolled_embeddings.npy",
            new_enrolled_embeddings,
            self.storage_dir / "employee_enrolled_uuids.npy",
            new_enrolled_uuids,
        )
        self.enrolled_embeddings = new_enrolled_embeddings
        self.enrolled_uuids = new_enrolled_uuids
        self.embeddings = np.ascontiguousarray(new_embeddings, dtype=np.float32)
        self.uuids = new_uuids
        self.index = candidate_index
        self.index_error = None
        return {
            "added": not updated,
            "updated": updated,
            "already_indexed": False,
            "vectors_count": int(self.index.ntotal),
            "employees_count": int(len(np.unique(self.uuids))),
        }

    @_index_synchronized
    def add_video_embeddings(
        self,
        employee_uuid: str,
        embeddings: list[np.ndarray],
    ) -> dict[str, Any]:
        if not employee_uuid:
            raise ArcFaceIndexError("UUID сотрудника не должен быть пустым.")
        if self.index is None or self.embeddings is None or self.uuids is None:
            raise RuntimeError("Индекс сотрудников пока не загружен.")
        if (
            self.enrolled_embeddings is None
            or self.enrolled_uuids is None
            or self.video_embeddings is None
            or self.video_uuids is None
        ):
            raise RuntimeError("Состояние дополнительных эмбеддингов не загружено.")

        base_count = (
            len(self.embeddings)
            - len(self.enrolled_embeddings)
            - len(self.video_embeddings)
        )
        known_for_employee = [
            vector.copy() for vector in self.embeddings[self.uuids == employee_uuid]
        ]
        new_vectors: list[np.ndarray] = []
        for candidate in embeddings:
            vector = np.asarray(candidate, dtype=np.float32).reshape(-1)
            if vector.shape[0] != self.index.d or not np.isfinite(vector).all():
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                continue
            vector = np.ascontiguousarray(vector / norm, dtype=np.float32)
            if known_for_employee and max(
                float(np.dot(known, vector)) for known in known_for_employee
            ) >= 0.99999:
                continue
            known_for_employee.append(vector)
            new_vectors.append(vector)

        if not new_vectors:
            return {
                "added_vectors": 0,
                "skipped_vectors": len(embeddings),
                "vectors_count": int(self.index.ntotal),
                "employees_count": int(len(np.unique(self.uuids))),
                "video_vectors_count": int(len(self.video_embeddings)),
            }

        new_video_embeddings = np.ascontiguousarray(
            np.concatenate(
                (self.video_embeddings, np.stack(new_vectors)),
                axis=0,
            ),
            dtype=np.float32,
        )
        new_video_uuids = np.concatenate(
            (
                self.video_uuids,
                np.asarray([employee_uuid] * len(new_vectors), dtype=str),
            )
        )
        new_embeddings = np.ascontiguousarray(
            np.concatenate(
                (
                    self.embeddings[:base_count],
                    self.enrolled_embeddings,
                    new_video_embeddings,
                ),
                axis=0,
            ),
            dtype=np.float32,
        )
        new_uuids = np.concatenate(
            (
                self.uuids[:base_count],
                self.enrolled_uuids,
                new_video_uuids,
            )
        )
        candidate_index = faiss.IndexFlatL2(int(new_embeddings.shape[1]))
        candidate_index.add(new_embeddings)
        _atomic_save_array_pair(
            self.storage_dir / "employee_video_embeddings.npy",
            new_video_embeddings,
            self.storage_dir / "employee_video_uuids.npy",
            new_video_uuids,
        )
        self.video_embeddings = new_video_embeddings
        self.video_uuids = new_video_uuids
        self.embeddings = new_embeddings
        self.uuids = new_uuids
        self.index = candidate_index
        self.index_error = None
        return {
            "added_vectors": len(new_vectors),
            "skipped_vectors": len(embeddings) - len(new_vectors),
            "vectors_count": int(self.index.ntotal),
            "employees_count": int(len(np.unique(self.uuids))),
            "video_vectors_count": int(len(self.video_embeddings)),
        }

    def add_employee_image(
        self,
        image_bytes: bytes,
        employee_uuid: str,
    ) -> dict[str, Any]:
        embedding = self.embedding_from_image(image_bytes)
        return self.add_embedding(employee_uuid, embedding)

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        embedding = self.embedding_from_image(image_bytes)
        return self.search_embedding(embedding, top_k=top_k, threshold=threshold)
