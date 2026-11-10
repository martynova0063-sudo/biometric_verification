import unittest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.drift_detection import (
    build_histogram,
    calculate_psi,
    run_data_drift_analysis,
    update_manual_review_mode,
)
from app.models import (
    DataDriftBaseline,
    DataDriftObservation,
    DataDriftRun,
    RegisteredModel,
    VerificationSession,
)
from app.threshold_manager import recalibrate_match_threshold


class DriftHistogramTests(unittest.TestCase):
    def test_identical_histograms_have_zero_psi(self):
        values = [0.12, 0.18, 0.22, 0.28, 0.31]
        histogram = build_histogram(values)
        self.assertAlmostEqual(calculate_psi(histogram, histogram), 0.0, places=9)

    def test_distribution_shift_crosses_critical_threshold(self):
        baseline = build_histogram([0.20] * 100)
        current = build_histogram([0.90] * 100)
        self.assertGreaterEqual(calculate_psi(baseline, current), 0.25)

    def test_empty_histogram_is_not_evaluable(self):
        self.assertIsNone(calculate_psi(build_histogram([]), build_histogram([0.2])))


class ThresholdCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def test_threshold_is_updated_only_with_minimum_label_counts(self):
        now = datetime(2026, 10, 21, 12, 0, 0)
        with self.session_factory() as db:
            model = RegisteredModel(
                model_name="HEALTH_ID",
                version="drift-test-1",
                status="current",
                purpose="test",
                change_summary="test",
                parameters={"match_threshold": 0.85},
            )
            db.add(model)
            created_at = now - timedelta(days=1)
            db.add_all([
                VerificationSession(
                    employee_ref=f"impostor-{index}",
                    status="not_verified",
                    match_score=0.20,
                    liveness_score=0.9,
                    quality_score=0.9,
                    decision="reject",
                    algorithm_version="test",
                    created_at=created_at,
                    audit_ref=f"AUD-{index}",
                    ground_truth_match=False,
                )
                for index in range(1_000)
            ])
            db.add_all([
                VerificationSession(
                    employee_ref=f"genuine-{index}",
                    status="verified",
                    match_score=0.80,
                    liveness_score=0.9,
                    quality_score=0.9,
                    decision="accept",
                    algorithm_version="test",
                    created_at=created_at,
                    audit_ref=f"GEN-{index}",
                    ground_truth_match=True,
                )
                for index in range(100)
            ])
            db.flush()

            result = recalibrate_match_threshold(db, now=now)

            self.assertEqual(result["status"], "applied")
            self.assertEqual(result["impostor_samples"], 1_000)
            self.assertEqual(result["genuine_samples"], 100)
            self.assertLess(result["achieved_far"], 0.001)
            self.assertGreater(result["proposed_threshold"], 0.20)
            self.assertLessEqual(result["proposed_threshold"], 0.80)
            self.assertEqual(model.parameters["match_threshold"], result["proposed_threshold"])

    def test_threshold_stays_fixed_during_first_fourteen_days(self):
        now = datetime(2026, 10, 21, 12, 0, 0)
        with self.session_factory() as db:
            model = RegisteredModel(
                model_name="HEALTH_ID",
                version="drift-initial-window",
                status="current",
                purpose="test",
                change_summary="test",
                parameters={"match_threshold": 0.85},
            )
            db.add(model)
            db.add(DataDriftObservation(
                source_type="verification",
                observed_at=now - timedelta(days=7),
                is_test=False,
                metrics={"match_score": 0.4},
                capture_metadata={},
            ))
            db.flush()
            result = run_data_drift_analysis(db, now=now)
            db.refresh(model)
            self.assertEqual(result["recalibration_status"], "initial_collection")
            self.assertEqual(model.parameters["match_threshold"], 0.85)

    def test_critical_drift_cannot_be_cleared_without_operator_review(self):
        now = datetime(2026, 10, 21, 12, 0, 0)
        with self.session_factory() as db:
            db.add(DataDriftRun(
                calculation_date=now.date(),
                window_start=now - timedelta(days=7),
                window_end=now,
                status="critical",
                psi_values={"match_score": 0.3},
                sample_counts={"match_score": 100},
                needs_recalibration=True,
                manual_review_mode=True,
                alert_messages=["critical"],
                recalibration={},
                created_at=now,
            ))
            db.flush()
            with self.assertRaises(ValueError):
                update_manual_review_mode(
                    db,
                    enabled=False,
                    reviewer_note="Reviewed",
                    now=now,
                )

    def test_new_camera_model_creates_recalibration_warning(self):
        now = datetime(2026, 10, 21, 12, 0, 0)
        baseline_end = now - timedelta(days=7)
        with self.session_factory() as db:
            db.add(DataDriftBaseline(
                id=1,
                window_start=now - timedelta(days=30),
                window_end=baseline_end,
                distributions={"match_score": build_histogram([0.5] * 100)},
                observation_count=100,
                created_at=baseline_end,
            ))
            db.add(DataDriftObservation(
                source_type="verification",
                observed_at=baseline_end - timedelta(days=1),
                is_test=False,
                metrics={"match_score": 0.5},
                capture_metadata={"camera_model": "Baseline Camera"},
            ))
            db.add_all([
                DataDriftObservation(
                    source_type="verification",
                    observed_at=now - timedelta(days=1),
                    is_test=False,
                    metrics={"match_score": 0.5},
                    capture_metadata={"camera_model": "New Camera"},
                )
                for _ in range(100)
            ])
            db.flush()

            result = run_data_drift_analysis(db, now=now)

            self.assertEqual(result["status"], "warning")
            self.assertTrue(result["needs_recalibration"])
            self.assertTrue(any("New Camera" in alert for alert in result["alert_messages"]))


if __name__ == "__main__":
    unittest.main()
