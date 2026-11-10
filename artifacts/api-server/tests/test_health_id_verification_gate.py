import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import calculate_health_id, list_eligible_health_employees
from app.models import Employee, ReviewTask, VerificationSession
from app.schemas import HealthCalculationInput, HealthMeasurements, HealthResultResponse


class HealthIdVerificationGateTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def add_employee(self, db, employee_ref):
        employee = Employee(
            guid=str(uuid4()),
            personnel_number=employee_ref,
            sex="",
            surname=f"Сотрудник-{employee_ref}",
            given_name="Тест",
            patronymic="",
            birth_date="",
            organization="Тестовая организация",
            is_test=False,
        )
        db.add(employee)
        db.flush()
        return employee

    def add_verification(
        self,
        db,
        employee_ref,
        *,
        status,
        decision,
        created_at=None,
        ground_truth_match=None,
        ground_truth_liveness=None,
    ):
        session = VerificationSession(
            id=str(uuid4()),
            employee_ref=employee_ref,
            status=status,
            match_score=0.9,
            liveness_score=0.9,
            quality_score=0.9,
            decision=decision,
            algorithm_version="test-model",
            created_at=created_at or datetime(2026, 10, 25, 10),
            audit_ref=f"AUD-{employee_ref}",
            details={"review_tasks": []},
            is_match=True,
            is_live=True,
            ground_truth_match=ground_truth_match,
            ground_truth_liveness=ground_truth_liveness,
        )
        db.add(session)
        db.flush()
        return session

    def add_review_task(
        self,
        db,
        session,
        *,
        resolution,
        status="resolved",
        resolved_at=None,
    ):
        task = ReviewTask(
            kind="verification",
            subject_ref=session.employee_ref,
            status=status,
            priority="high",
            summary="Ручная проверка",
            resolved_at=resolved_at or datetime(2026, 10, 25, 11),
            resolution=resolution,
        )
        task.verification_session_id = session.id
        db.add(task)
        db.flush()
        return task

    def test_eligible_employee_list_uses_latest_success_or_manual_approval(self):
        with self.session_factory() as db:
            self.add_employee(db, "EMP-SYSTEM")
            self.add_verification(
                db,
                "EMP-SYSTEM",
                status="verified",
                decision="match",
            )

            self.add_employee(db, "EMP-MANUAL")
            manual_session = self.add_verification(
                db,
                "EMP-MANUAL",
                status="manual_review",
                decision="review",
                ground_truth_match=False,
                ground_truth_liveness=False,
            )
            self.add_review_task(db, manual_session, resolution="approve")

            self.add_employee(db, "EMP-REJECTED")
            rejected_session = self.add_verification(
                db,
                "EMP-REJECTED",
                status="verified",
                decision="match",
            )
            self.add_review_task(db, rejected_session, resolution="reject")

            self.add_employee(db, "EMP-STALE")
            self.add_verification(
                db,
                "EMP-STALE",
                status="verified",
                decision="match",
                created_at=datetime(2026, 10, 25, 9),
            )
            self.add_verification(
                db,
                "EMP-STALE",
                status="insufficient_quality",
                decision="review",
                created_at=datetime(2026, 10, 25, 12),
            )
            db.commit()

            eligible = list_eligible_health_employees(db)

            self.assertEqual(
                {item["employee_ref"] for item in eligible},
                {"EMP-SYSTEM", "EMP-MANUAL"},
            )
            by_ref = {item["employee_ref"]: item for item in eligible}
            self.assertEqual(by_ref["EMP-SYSTEM"]["eligibility_source"], "system")
            self.assertEqual(by_ref["EMP-MANUAL"]["eligibility_source"], "manual")
            self.assertEqual(
                by_ref["EMP-MANUAL"]["verified_at"],
                datetime(2026, 10, 25, 11),
            )

    def test_calculation_rejects_employee_without_qualifying_latest_verification(self):
        with self.session_factory() as db:
            self.add_employee(db, "EMP-PENDING")
            self.add_verification(
                db,
                "EMP-PENDING",
                status="manual_review",
                decision="review",
            )
            db.commit()

            payload = HealthCalculationInput(
                employee_ref="EMP-PENDING",
                event_date=datetime(2026, 10, 25),
                measurements=HealthMeasurements(
                    sleep_hours=7,
                    stress_score=2,
                    activity_minutes=30,
                    resting_pulse=68,
                    respiratory_rate=15,
                    verification_quality=0.9,
                ),
            )
            with patch("app.main.calculate_health") as calculate:
                with self.assertRaises(HTTPException) as error:
                    calculate_health_id(payload, db)

            self.assertEqual(error.exception.status_code, 409)
            calculate.assert_not_called()

    def test_calculation_accepts_manually_approved_session_even_with_negative_labels(self):
        with self.session_factory() as db:
            self.add_employee(db, "EMP-MANUAL")
            manual_session = self.add_verification(
                db,
                "EMP-MANUAL",
                status="manual_review",
                decision="review",
                ground_truth_match=False,
                ground_truth_liveness=False,
            )
            self.add_review_task(db, manual_session, resolution="approve")
            db.commit()

            payload = HealthCalculationInput(
                employee_ref="EMP-MANUAL",
                event_date=datetime(2026, 10, 25),
                measurements=HealthMeasurements(
                    sleep_hours=7,
                    stress_score=2,
                    activity_minutes=30,
                    resting_pulse=68,
                    respiratory_rate=15,
                    verification_quality=0.9,
                ),
            )
            result_data = {
                "health_id": 0.75,
                "category": "stable",
                "status": "calculated",
                "components": {},
                "evidence": [],
            }
            with (
                patch("app.main.calculate_health", return_value=result_data),
                patch("app.main.record_drift_observation"),
            ):
                result = calculate_health_id(payload, db)

            self.assertEqual(result.employee_ref, "EMP-MANUAL")
            self.assertEqual(result.health_id, 0.75)
            self.assertEqual(
                result.measurements,
                payload.measurements.model_dump(exclude={"verification_quality"}),
            )
            self.assertNotIn("measurements", HealthResultResponse.model_validate(result).model_dump())


if __name__ == "__main__":
    unittest.main()
