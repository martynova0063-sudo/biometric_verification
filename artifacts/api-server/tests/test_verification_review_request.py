import unittest
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import request_verification_review, resolve_review_task, verification_session_payload
from app.models import ReviewTask, VerificationSession
from app.schemas import ReviewInput, VerificationSessionResponse


class ManualReviewRequestTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def test_request_queues_verified_match_without_changing_system_decision(self):
        session_id = uuid4()
        with self.session_factory() as db:
            session = VerificationSession(
                id=str(session_id),
                employee_ref="EMP-100",
                status="verified",
                match_score=0.96,
                liveness_score=0.92,
                quality_score=0.94,
                decision="match",
                algorithm_version="test-model",
                audit_ref="AUD-TEST-100",
                is_match=True,
                is_live=True,
                cosine_distance=0.96,
                details={"review_tasks": []},
            )
            db.add(session)
            db.commit()

            first_task = request_verification_review(session_id, db)
            second_task = request_verification_review(session_id, db)
            db.refresh(session)

            self.assertEqual(first_task.id, second_task.id)
            self.assertEqual(first_task.status, "open")
            self.assertEqual(first_task.kind, "verification")
            self.assertEqual(first_task.verification_session_id, str(session_id))
            self.assertIn("решение системы сохранено без изменений", first_task.summary)
            self.assertEqual(session.status, "verified")
            self.assertEqual(session.decision, "match")
            self.assertTrue(session.is_match)
            self.assertEqual(session.details["review_tasks"][0]["status"], "open")
            self.assertEqual(
                db.scalar(select(func.count()).select_from(ReviewTask)),
                1,
            )

            resolve_review_task(
                UUID(first_task.id),
                ReviewInput(
                    resolution="approve",
                    reviewer_note="Проверено специалистом.",
                    ground_truth_match=True,
                    ground_truth_liveness=True,
                ),
                db,
            )
            db.refresh(session)
            self.assertEqual(session.status, "verified")
            self.assertEqual(session.decision, "match")
            self.assertTrue(session.is_match)
            self.assertTrue(session.ground_truth_match)
            self.assertEqual(session.details["review_tasks"][0]["status"], "resolved")

    def test_request_rejects_sessions_without_automatic_match(self):
        session_id = uuid4()
        with self.session_factory() as db:
            session = VerificationSession(
                id=str(session_id),
                employee_ref="EMP-101",
                status="not_verified",
                match_score=0.12,
                liveness_score=0.92,
                quality_score=0.94,
                decision="no_match",
                algorithm_version="test-model",
                audit_ref="AUD-TEST-101",
                is_match=False,
                is_live=True,
                cosine_distance=0.12,
                details={"review_tasks": []},
            )
            db.add(session)
            db.commit()

            with self.assertRaises(HTTPException) as error:
                request_verification_review(session_id, db)

            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(session.status, "not_verified")
            self.assertEqual(session.decision, "no_match")
            self.assertEqual(
                db.scalar(select(func.count()).select_from(ReviewTask)),
                0,
            )

    def test_verification_duration_is_returned_and_legacy_sessions_remain_empty(self):
        with self.session_factory() as db:
            session = VerificationSession(
                employee_ref="EMP-102",
                status="verified",
                match_score=0.91,
                liveness_score=0.88,
                quality_score=0.93,
                decision="match",
                algorithm_version="test-model",
                audit_ref="AUD-TEST-102",
                is_match=True,
                is_live=True,
                cosine_distance=0.91,
                details={"verification_elapsed_seconds": 4.275},
            )
            db.add(session)
            db.commit()

            payload = verification_session_payload(session)
            response = VerificationSessionResponse.model_validate(payload)
            self.assertEqual(response.verification_elapsed_seconds, 4.275)

            session.details = {}
            legacy_response = VerificationSessionResponse.model_validate(
                verification_session_payload(session)
            )
            self.assertIsNone(legacy_response.verification_elapsed_seconds)


if __name__ == "__main__":
    unittest.main()
