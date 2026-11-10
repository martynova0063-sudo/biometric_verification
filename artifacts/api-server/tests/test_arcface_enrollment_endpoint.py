import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from fastapi import HTTPException

from app import main


class ArcFaceEnrollmentEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_enrolls_the_saved_private_reference_photo(self):
        employee_guid = str(uuid4())
        employee = SimpleNamespace(
            guid=employee_guid,
            photo_path=f"/objects/employees/{employee_guid}/reference.jpg",
            personnel_number="0001",
        )
        db = Mock()
        db.get.return_value = employee
        expected_result = {
            "added": True,
            "updated": False,
            "already_indexed": False,
            "vectors_count": 5,
            "employees_count": 4,
        }

        async def run_inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        with (
            patch.object(main, "employee_has_photo", return_value=True),
            patch.object(main, "is_private_employee_photo", return_value=True),
            patch.object(
                main,
                "fetch_private_employee_photo",
                new=AsyncMock(return_value=b"saved-jpeg-bytes"),
            ),
            patch.object(main, "run_in_threadpool", side_effect=run_inline),
            patch.object(
                main.arcface_search,
                "add_employee_image",
                return_value=expected_result,
            ) as add_image,
        ):
            result = await main.add_employee_to_arcface_index(
                UUID(employee_guid),
                db,
            )

        self.assertEqual(result["personnel_number"], "0001")
        self.assertEqual(result["vectors_count"], 5)
        self.assertEqual(
            result["message"],
            "Эмбеддинг эталонного фото сотрудника добавлен в индекс.",
        )
        add_image.assert_called_once_with(b"saved-jpeg-bytes", employee_guid)

    async def test_returns_not_found_when_employee_has_no_saved_photo(self):
        employee_guid = str(uuid4())
        db = Mock()
        db.get.return_value = SimpleNamespace(
            guid=employee_guid,
            photo_path=None,
            personnel_number="0001",
        )

        with patch.object(main, "employee_has_photo", return_value=False):
            with self.assertRaises(HTTPException) as raised:
                await main.add_employee_to_arcface_index(UUID(employee_guid), db)

        self.assertEqual(raised.exception.status_code, 404)
        self.assertIn("нет сохранённого", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
