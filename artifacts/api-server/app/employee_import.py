import json
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from .models import Employee


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMPLOYEE_SOURCE = DATA_DIR / "employee.json"
EMPLOYEE_PHOTOS_DIR = DATA_DIR / "employees"
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def employee_photo_path(photo_path: str | None) -> Path | None:
    if not photo_path:
        return None

    root = DATA_DIR.resolve()
    resolved = (root / photo_path).resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved


def is_private_employee_photo(photo_path: str | None, employee_guid: str) -> bool:
    prefix = f"/objects/employees/{employee_guid}/"
    if not photo_path or not photo_path.startswith(prefix):
        return False
    filename = photo_path[len(prefix):]
    return (
        bool(filename)
        and "/" not in filename
        and "\\" not in filename
        and filename.endswith(".jpg")
        and ".." not in filename
    )


def employee_has_photo(employee: Employee) -> bool:
    if is_private_employee_photo(employee.photo_path, employee.guid):
        return True
    photo_path = employee_photo_path(employee.photo_path)
    return bool(photo_path and photo_path.is_file())


def sync_employees(db: Session) -> int:
    if not EMPLOYEE_SOURCE.is_file():
        raise RuntimeError(f"Employee source data is missing: {EMPLOYEE_SOURCE}")

    records = json.loads(EMPLOYEE_SOURCE.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Employee source must contain a JSON array.")

    seen_guids: set[str] = set()
    seen_personnel_numbers: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Each employee record must be a JSON object.")

        guid = str(UUID(str(row["guid"])))
        personnel_number = str(row["Табельный номер"]).strip()
        if not personnel_number:
            raise ValueError(f"Employee {guid} has an empty personnel number.")
        if guid in seen_guids or personnel_number in seen_personnel_numbers:
            raise ValueError("Employee source contains duplicate GUIDs or personnel numbers.")
        seen_guids.add(guid)
        seen_personnel_numbers.add(personnel_number)

        required_fields = (
            "Пол",
            "Фамилия",
            "Имя",
            "Отчество",
            "Дата рождения",
            "Организация",
        )
        if any(not str(row.get(field, "")).strip() for field in required_fields):
            raise ValueError(f"Employee {guid} is missing a required field.")

        photo_dir = EMPLOYEE_PHOTOS_DIR / guid
        photos = sorted(
            path for path in photo_dir.iterdir()
            if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES
        ) if photo_dir.is_dir() else []
        relative_photo_path = (
            (Path("employees") / guid / photos[0].name).as_posix()
            if photos else None
        )

        employee = db.get(Employee, guid)
        if employee is None:
            employee = Employee(guid=guid)
            db.add(employee)

        employee.personnel_number = personnel_number
        employee.sex = str(row["Пол"]).strip()
        employee.surname = str(row["Фамилия"]).strip()
        employee.given_name = str(row["Имя"]).strip()
        employee.patronymic = str(row["Отчество"]).strip()
        employee.birth_date = str(row["Дата рождения"]).strip()
        employee.organization = str(row["Организация"]).strip()
        employee.is_photo = bool(row.get("is_photo", 0))
        employee.is_test = bool(row.get("is_test", 0))
        if relative_photo_path:
            employee.photo_path = relative_photo_path

    db.commit()
    return len(records)
