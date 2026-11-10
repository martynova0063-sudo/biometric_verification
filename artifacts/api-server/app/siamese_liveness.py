from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FaceEncoder(nn.Module):
    """Face encoder matching the supplied Siamese liveness checkpoint."""

    def __init__(self, input_channels: int = 3, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(self.features(image))
        return F.normalize(embedding, p=2, dim=1)


class SiameseAntiSpoofing(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = FaceEncoder(input_channels=3, embedding_dim=embedding_dim)

    def forward(
        self,
        probe: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_embedding = self.encoder(probe)
        reference_embedding = self.encoder(reference)
        distance = F.pairwise_distance(probe_embedding, reference_embedding)
        live_score = (1.0 - distance / 2.0).clamp(0.0, 1.0)
        return distance, live_score


_MODEL_CACHE: dict[tuple[str, int, int], SiameseAntiSpoofing] = {}
_MODEL_LOCK = Lock()


def load_siamese_model(checkpoint_path: Path) -> SiameseAntiSpoofing:
    stat = checkpoint_path.stat()
    cache_key = (str(checkpoint_path), stat.st_mtime_ns, stat.st_size)
    with _MODEL_LOCK:
        cached_model = _MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

        # weights_only avoids executing arbitrary objects from a checkpoint file.
        checkpoint: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Siamese checkpoint must contain a state dictionary.")

        model = SiameseAntiSpoofing()
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model


def prepare_face(face_bgr: np.ndarray) -> np.ndarray:
    """Apply the same RGB resize and ImageNet normalization used in training."""
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Cannot prepare an empty face crop.")
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    image = resized.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)


def score_face_pairs(
    model: SiameseAntiSpoofing,
    reference: np.ndarray,
    probes: list[np.ndarray],
    batch_size: int = 16,
) -> list[float]:
    if not probes:
        return []

    reference_tensor = torch.from_numpy(reference).unsqueeze(0)
    scores: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(probes), batch_size):
            batch = torch.from_numpy(np.stack(probes[offset : offset + batch_size]))
            batch_reference = reference_tensor.expand(batch.shape[0], -1, -1, -1)
            _distance, live_scores = model(batch, batch_reference)
            scores.extend(float(value) for value in live_scores.cpu().tolist())
    return scores
