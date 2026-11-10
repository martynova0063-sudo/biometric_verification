import unittest
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import get_personal_health_baseline, list_health_results
from app.models import Employee, HealthResult
from app.personal_baseline import calculate_personal_baseline
from app.schemas import HealthResultListResponse, PersonalBaselineResponse


class PersonalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def add_result(self, db, employee_ref, event_date, health_id, measurements=None):
        result = HealthResult(
            id=str(uuid4()),
            employee_ref=employee_ref,
            event_date=event_date,
            health_id=health_id,
            category="stable",
            status="calculated",
            model_version="test-model",
            components={},
            measurements=measurements,
            evidence=[],
            disclaimer="Research use only.",
        )
        db.add(result)
        db.flush()
        return result

    def test_uses_last_30_days_and_calculates_median_iqr_corridors(self):
        now = datetime(2026, 10, 26, 12)
        samples = [
            (0.2, 60, 4),
            (0.4, 70, 6),
            (0.6, 80, 8),
            (0.8, 90, 10),
        ]
        with self.session_factory() as db:
            for index, (health_id, pulse, sleep) in enumerate(samples):
                self.add_result(
                    db,
                    "EMP-1",
                    now - timedelta(days=index + 1),
                    health_id,
                    {
                        "resting_pulse": pulse,
                        "sleep_hours": sleep,
                        "stress_score": 4,
                        "activity_minutes": 30 + index,
                        "respiratory_rate": 14 + index,
                    },
                )
            self.add_result(
                db,
                "EMP-1",
                now - timedelta(days=31),
                0.99,
                {"resting_pulse": 230, "sleep_hours": 23},
            )
            self.add_result(
                db,
                "EMP-1",
                now + timedelta(days=1),
                0.99,
                {"resting_pulse": 230, "sleep_hours": 23},
            )
            self.add_result(
                db,
                "EMP-OTHER",
                now - timedelta(days=1),
                0.99,
                {"resting_pulse": 230, "sleep_hours": 23},
            )
            db.commit()

            baseline = calculate_personal_baseline(db, "EMP-1", now=now)

        by_key = {metric["key"]: metric for metric in baseline["metrics"]}
        self.assertEqual(baseline["window_days"], 30)
        self.assertEqual(len(by_key), 6)
        self.assertEqual(by_key["health_id"]["sample_count"], 4)
        self.assertAlmostEqual(by_key["health_id"]["median"], 0.5)
        self.assertAlmostEqual(by_key["health_id"]["lower_bound"], 0.05)
        self.assertAlmostEqual(by_key["health_id"]["upper_bound"], 0.95)
        self.assertEqual(by_key["resting_pulse"]["sample_count"], 4)
        self.assertAlmostEqual(by_key["resting_pulse"]["median"], 75)
        self.assertAlmostEqual(by_key["resting_pulse"]["lower_bound"], 52.5)
        self.assertAlmostEqual(by_key["resting_pulse"]["upper_bound"], 97.5)
        self.assertAlmostEqual(by_key["sleep_hours"]["lower_bound"], 2.5)
        self.assertAlmostEqual(by_key["sleep_hours"]["upper_bound"], 11.5)

    def test_old_results_without_measurements_show_insufficient_data(self):
        now = datetime(2026, 10, 26, 12)
        with self.session_factory() as db:
            for index in range(4):
                self.add_result(
                    db,
                    "EMP-OLD",
                    now - timedelta(days=index + 1),
                    0.4 + index * 0.1,
                )
            db.commit()

            baseline = calculate_personal_baseline(db, "EMP-OLD", now=now)

        by_key = {metric["key"]: metric for metric in baseline["metrics"]}
        self.assertTrue(by_key["health_id"]["ready"])
        self.assertEqual(by_key["health_id"]["sample_count"], 4)
        self.assertFalse(by_key["resting_pulse"]["ready"])
        self.assertEqual(by_key["resting_pulse"]["sample_count"], 0)
        self.assertIsNone(by_key["resting_pulse"]["lower_bound"])
        self.assertIsNone(by_key["resting_pulse"]["upper_bound"])

    def test_api_requires_an_employee_and_returns_typed_baseline(self):
        with self.session_factory() as db:
            with self.assertRaises(HTTPException) as error:
                get_personal_health_baseline("MISSING", db)
            self.assertEqual(error.exception.status_code, 404)

            db.add(Employee(
                guid=str(uuid4()),
                personnel_number="EMP-BASELINE",
                sex="",
                surname="Тестовый",
                given_name="Сотрудник",
                patronymic="",
                birth_date="",
                organization="Тест",
                is_test=False,
            ))
            db.commit()

            response = get_personal_health_baseline("EMP-BASELINE", db)
            validated = PersonalBaselineResponse.model_validate(response)

        self.assertEqual(validated.employee_ref, "EMP-BASELINE")
        self.assertEqual(validated.window_days, 30)
        self.assertEqual(len(validated.metrics), 6)
        self.assertTrue(all(metric.sample_count == 0 for metric in validated.metrics))

    def test_results_list_marks_values_outside_prior_personal_corridor(self):
        event_date = datetime(2026, 10, 27, 12)
        with self.session_factory() as db:
            for index, (health_id, pulse) in enumerate([
                (0.40, 60),
                (0.42, 61),
                (0.44, 62),
                (0.46, 63),
            ]):
                self.add_result(
                    db,
                    "EMP-OUTLIER",
                    event_date - timedelta(days=index + 1),
                    health_id,
                    {"resting_pulse": pulse},
                )
            current = self.add_result(
                db,
                "EMP-OUTLIER",
                event_date,
                0.9,
                {"resting_pulse": 150},
            )
            db.commit()

            result_rows = list_health_results(category="all", limit=100, db=db)

        row = next(item for item in result_rows if str(item["id"]) == current.id)
        validated = HealthResultListResponse.model_validate(row)
        self.assertEqual(validated.personal_baseline.status, "outside_corridor")
        self.assertEqual(
            set(validated.personal_baseline.outside_metrics),
            {"health_id", "resting_pulse"},
        )
        self.assertEqual(validated.personal_baseline.evaluated_metric_count, 2)
        self.assertNotIn("measurements", row)


if __name__ == "__main__":
    unittest.main()
