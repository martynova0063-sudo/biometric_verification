"""Rebuild the employee vector index from data/arcface_dataset.

Adding an employee is an enrollment operation: add their UUID folder and photos
to the dataset, then run this script to regenerate the vectors. The trained
ArcFace checkpoint remains unchanged; this script does not fine-tune its weights.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .arcface_model import ResNet50ArcFace


API_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npy", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.save(temporary_path, value, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, value: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_embeddings(
    *,
    model_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int = 16,
) -> dict[str, int]:
    if batch_size < 1:
        raise ValueError("batch_size должен быть больше нуля.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Папка с фото не найдена: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict):
        raise ValueError("В checkpoint отсутствует class_to_idx.")

    model = ResNet50ArcFace(
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    embeddings: list[np.ndarray] = []
    employee_uuids: list[str] = []
    skipped_images = 0
    image_paths = [
        (employee_dir.name, image_path)
        for employee_dir in sorted(data_root.iterdir())
        if employee_dir.is_dir()
        for image_path in sorted(employee_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"В папке {data_root} нет подходящих фотографий.")

    for start in range(0, len(image_paths), batch_size):
        batch_images: list[torch.Tensor] = []
        batch_uuids: list[str] = []
        for employee_uuid, image_path in image_paths[start:start + batch_size]:
            try:
                with Image.open(image_path) as source:
                    batch_images.append(IMAGE_TRANSFORM(source.convert("RGB")))
                batch_uuids.append(employee_uuid)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                skipped_images += 1
                print(f"Пропущен файл {image_path}: {error}")

        if not batch_images:
            continue
        inputs = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            batch_embeddings = model(inputs).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        if np.any(norms <= 1e-12) or not np.isfinite(batch_embeddings).all():
            raise ValueError("Модель вернула нулевые или некорректные эмбеддинги.")
        embeddings.extend((batch_embeddings / norms).astype(np.float32))
        employee_uuids.extend(batch_uuids)

    if not embeddings:
        raise ValueError("Не удалось построить ни одного эмбеддинга.")

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_array = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    uuid_array = np.asarray(employee_uuids, dtype=str)
    index_to_uuid = {
        str(index): employee_uuid
        for employee_uuid, index in sorted(
            ((str(uuid), int(index)) for uuid, index in class_to_idx.items()),
            key=lambda item: item[1],
        )
    }

    # UUID rows are the authoritative mapping for the search index. The JSON
    # retains the classifier's training-class mapping and need not contain new
    # employees enrolled only in the vector index.
    _atomic_save_npy(output_dir / "employee_embeddings.npy", embedding_array)
    _atomic_save_npy(output_dir / "employee_uuids.npy", uuid_array)
    _atomic_save_json(output_dir / "class_mapping.json", index_to_uuid)

    result = {
        "vectors_count": int(embedding_array.shape[0]),
        "employees_count": int(len(set(employee_uuids))),
        "dim": int(embedding_array.shape[1]),
        "skipped_images": skipped_images,
    }
    print(
        f"Готово: {result['vectors_count']} векторов, "
        f"{result['employees_count']} сотрудников, "
        f"пропущено фото: {skipped_images}. Индекс сохранён в {output_dir}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить/обновить базу эмбеддингов ArcFace."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=API_ROOT / "app/storage/best_arcface_resnet50.pth",
        help="Путь к checkpoint ArcFace.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=API_ROOT / "data/arcface_dataset",
        help="Путь к папкам с фотографиями сотрудников (папка = UUID).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=API_ROOT / "app/storage",
        help="Каталог для employee_embeddings.npy, employee_uuids.npy и class_mapping.json.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    generate_embeddings(
        model_path=args.model,
        data_root=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
