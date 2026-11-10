import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.main import request_employee_photo_upload
from app.employee_photo_archive import (
    EmployeePhotoArchiveError,
    archive_current_employee_photo,
)
from app.models import Employee, EmployeePhotoArchive, EmployeePhotoUpload
from app.object_storage import ObjectStorageError
from app.schemas import EmployeePhotoUploadUrlInput


class FakeSession:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)

    def get(self, model, key):
        return self.employee if model is Employee else None

    def execute(self, statement):
        return None

    def commit(self):
        return None


class EmployeePhotoArchiveTests(unittest.TestCase):
    def setUp(self):
        self.guid = str(uuid4())
        self.source_path = f"/objects/employees/{self.guid}/current.jpg"
        self.employee = Employee(guid=self.guid, photo_path=self.source_path)
        self.db = FakeSession()
        self.photo_bytes = b"\xff\xd8\xffprevious-photo"

    def test_copies_existing_private_photo_to_dataset_and_keeps_current_reference(self):
        with (
            patch(
                "app.employee_photo_archive.fetch_private_employee_photo",
                new=AsyncMock(return_value=self.photo_bytes),
            ),
            patch(
                "app.employee_photo_archive.store_dataset_employee_photo",
                new=AsyncMock(return_value=f"/objects/dataset/employees/{self.guid}/archived.jpg"),
            ) as store_photo,
        ):
            archive = asyncio.run(
                archive_current_employee_photo(self.employee, "operator-1", self.db)
            )

        self.assertIsInstance(archive, EmployeePhotoArchive)
        self.assertEqual(self.employee.photo_path, self.source_path)
        self.assertEqual(archive.source_photo_path, self.source_path)
        self.assertEqual(
            archive.object_path,
            f"/objects/dataset/employees/{self.guid}/archived.jpg",
        )
        self.assertEqual(archive.archived_by, "operator-1")
        self.assertEqual(archive.size_bytes, len(self.photo_bytes))
        self.assertEqual(len(archive.sha256), 64)
        self.assertEqual(self.db.records, [archive])
        store_photo.assert_awaited_once_with(
            self.guid,
            ".jpg",
            "image/jpeg",
            self.photo_bytes,
        )

    def test_storage_failure_does_not_create_archive_record_or_change_current_photo(self):
        with (
            patch(
                "app.employee_photo_archive.fetch_private_employee_photo",
                new=AsyncMock(return_value=self.photo_bytes),
            ),
            patch(
                "app.employee_photo_archive.store_dataset_employee_photo",
                new=AsyncMock(side_effect=ObjectStorageError("storage unavailable")),
            ),
        ):
            with self.assertRaises(EmployeePhotoArchiveError):
                asyncio.run(
                    archive_current_employee_photo(self.employee, "operator-1", self.db)
                )

        self.assertEqual(self.employee.photo_path, self.source_path)
        self.assertEqual(self.db.records, [])

    def test_copies_original_local_photo_into_dataset_without_deleting_it(self):
        with TemporaryDirectory() as directory:
            original_file = Path(directory) / "portrait.png"
            original_bytes = b"\x89PNG\r\n\x1a\noriginal-photo"
            original_file.write_bytes(original_bytes)
            employee = Employee(
                guid=self.guid,
                photo_path=f"employees/{self.guid}/portrait.png",
            )

            with (
                patch(
                    "app.employee_photo_archive.employee_has_photo",
                    return_value=True,
                ),
                patch(
                    "app.employee_photo_archive.employee_photo_path",
                    return_value=original_file,
                ),
                patch(
                    "app.employee_photo_archive.store_dataset_employee_photo",
                    new=AsyncMock(
                        return_value=f"/objects/dataset/employees/{self.guid}/archived.png"
                    ),
                ) as store_photo,
            ):
                archive = asyncio.run(
                    archive_current_employee_photo(employee, "operator-1", self.db)
                )

            self.assertEqual(archive.content_type, "image/png")
            self.assertEqual(archive.source_photo_path, employee.photo_path)
            self.assertTrue(original_file.is_file())
            self.assertEqual(original_file.read_bytes(), original_bytes)
            store_photo.assert_awaited_once_with(
                self.guid,
                ".png",
                "image/png",
                original_bytes,
            )

    def test_upload_url_can_be_created_when_employee_already_has_a_photo(self):
        self.db.employee = self.employee
        request = SimpleNamespace(
            state=SimpleNamespace(auth_user={"id": "operator-1"})
        )
        object_path = f"/objects/employees/{self.guid}/replacement.jpg"

        with patch(
            "app.main.sign_employee_photo_upload",
            new=AsyncMock(return_value=("https://storage.example/upload", object_path)),
        ):
            result = asyncio.run(
                request_employee_photo_upload(
                    self.guid,
                    EmployeePhotoUploadUrlInput(size=100, content_type="image/jpeg"),
                    request,
                    self.db,
                )
            )

        self.assertEqual(result["object_path"], object_path)
        self.assertTrue(any(isinstance(record, EmployeePhotoUpload) for record in self.db.records))
