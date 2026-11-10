from datetime import datetime
from hashlib import sha256
from mimetypes import guess_type

from sqlalchemy.orm import Session

from .employee_import import employee_has_photo, employee_photo_path, is_private_employee_photo
from .models import Employee, EmployeePhotoArchive
from .object_storage import (
    MAX_ARCHIVE_PHOTO_BYTES,
    ObjectStorageError,
    fetch_private_employee_photo,
    store_dataset_employee_photo,
)


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class EmployeePhotoArchiveError(Exception):
    pass


async def archive_current_employee_photo(
    employee: Employee,
    archived_by: str,
    db: Session,
) -> EmployeePhotoArchive | None:
    source_path = employee.photo_path
    if not employee_has_photo(employee) or not source_path:
        return None

    if is_private_employee_photo(source_path, employee.guid):
        extension = ".jpg"
        content_type = "image/jpeg"
        try:
            photo_bytes = await fetch_private_employee_photo(source_path)
        except ObjectStorageError as error:
            raise EmployeePhotoArchiveError("Could not read the current reference photo.") from error
        if not photo_bytes:
            raise EmployeePhotoArchiveError("The current reference photo is missing from storage.")
    else:
        source_file = employee_photo_path(source_path)
        if not source_file or not source_file.is_file():
            raise EmployeePhotoArchiveError("The current reference photo is not available to archive.")
        extension = source_file.suffix.lower()
        content_type = guess_type(source_file.name)[0]
        if extension not in ALLOWED_EXTENSIONS or content_type not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise EmployeePhotoArchiveError("The current reference photo has an unsupported format.")
        try:
            if source_file.stat().st_size > MAX_ARCHIVE_PHOTO_BYTES:
                raise EmployeePhotoArchiveError("The current reference photo exceeds the archive size limit.")
            photo_bytes = source_file.read_bytes()
        except OSError as error:
            raise EmployeePhotoArchiveError("Could not read the current reference photo.") from error

    if not photo_bytes or len(photo_bytes) > MAX_ARCHIVE_PHOTO_BYTES:
        raise EmployeePhotoArchiveError("The current reference photo exceeds the archive size limit.")

    try:
        archived_path = await store_dataset_employee_photo(
            employee.guid,
            extension,
            content_type,
            photo_bytes,
        )
    except ObjectStorageError as error:
        raise EmployeePhotoArchiveError("Could not store the current reference photo in the dataset.") from error

    archive = EmployeePhotoArchive(
        employee_guid=employee.guid,
        source_photo_path=source_path,
        object_path=archived_path,
        archived_by=archived_by,
        content_type=content_type,
        sha256=sha256(photo_bytes).hexdigest(),
        size_bytes=len(photo_bytes),
        created_at=datetime.utcnow(),
    )
    db.add(archive)
    return archive
