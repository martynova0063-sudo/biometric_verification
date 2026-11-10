from datetime import datetime, timedelta, timezone
import os
from typing import Literal
from uuid import uuid4

import httpx


SIDECAR_URL = "http://127.0.0.1:1106"
MAX_PHOTO_BYTES = 5 * 1024 * 1024
MAX_ARCHIVE_PHOTO_BYTES = 25 * 1024 * 1024


class ObjectStorageError(Exception):
    pass


def _private_object_dir() -> str:
    path = os.getenv("PRIVATE_OBJECT_DIR", "").strip().rstrip("/")
    if not path:
        raise ObjectStorageError("Private object storage is not configured.")
    return path


def _bucket_and_name(path: str) -> tuple[str, str]:
    parts = path.strip("/").split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ObjectStorageError("Invalid private object path.")
    return parts[0], "/".join(parts[1:])


async def sign_employee_photo_upload(
    employee_guid: str,
) -> tuple[str, str]:
    private_dir = _private_object_dir()
    object_name_suffix = f"employees/{employee_guid}/{uuid4().hex}"
    object_path = f"/objects/{object_name_suffix}.jpg"
    bucket_name, object_name = _bucket_and_name(
        f"{private_dir}/{object_name_suffix}.jpg"
    )
    upload_url = await _sign_object_url(bucket_name, object_name, "PUT", 15 * 60)
    return upload_url, object_path


async def fetch_private_employee_photo(object_path: str) -> bytes | None:
    if not object_path.startswith("/objects/employees/"):
        return None
    relative_path = object_path.removeprefix("/objects/")
    if any(part in {"", ".", ".."} for part in relative_path.split("/")):
        return None

    bucket_name, object_name = _bucket_and_name(
        f"{_private_object_dir()}/{relative_path}"
    )
    signed_url = await _sign_object_url(bucket_name, object_name, "GET", 60)

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            async with client.stream("GET", signed_url) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                if response.headers.get("content-type", "").split(";", 1)[0].lower() != "image/jpeg":
                    raise ObjectStorageError("Stored employee photo is not a JPEG image.")

                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_PHOTO_BYTES:
                        raise ObjectStorageError("Stored employee photo exceeds the size limit.")
                return bytes(content)
    except httpx.HTTPError as error:
        raise ObjectStorageError("Could not read the private employee photo.") from error


async def store_dataset_employee_photo(
    employee_guid: str,
    extension: str,
    content_type: str,
    photo_bytes: bytes,
) -> str:
    allowed_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    normalized_extension = extension.lower()
    if allowed_types.get(normalized_extension) != content_type:
        raise ObjectStorageError("Unsupported archived employee photo format.")
    if not photo_bytes or len(photo_bytes) > MAX_ARCHIVE_PHOTO_BYTES:
        raise ObjectStorageError("Archived employee photo exceeds the size limit.")

    object_name = f"dataset/employees/{employee_guid}/{uuid4().hex}{normalized_extension}"
    bucket_name, bucket_object_name = _bucket_and_name(
        f"{_private_object_dir()}/{object_name}"
    )
    signed_url = await _sign_object_url(bucket_name, bucket_object_name, "PUT", 15 * 60)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.put(
                signed_url,
                content=photo_bytes,
                headers={"Content-Type": content_type},
            )
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise ObjectStorageError("Could not archive the previous employee photo.") from error

    return f"/objects/{object_name}"


async def _sign_object_url(
    bucket_name: str,
    object_name: str,
    method: Literal["GET", "PUT", "DELETE", "HEAD"],
    ttl_seconds: int,
) -> str:
    request = {
        "bucket_name": bucket_name,
        "object_name": object_name,
        "method": method,
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        ).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{SIDECAR_URL}/object-storage/signed-object-url",
                json=request,
            )
            response.raise_for_status()
            signed_url = response.json().get("signed_url")
            if not isinstance(signed_url, str) or not signed_url.startswith("https://"):
                raise ObjectStorageError("The object storage service returned an invalid URL.")
            return signed_url
    except httpx.HTTPError as error:
        raise ObjectStorageError("Could not connect to private object storage.") from error
