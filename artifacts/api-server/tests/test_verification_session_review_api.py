import unittest
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import (
    get_verification_session_status,
    list_pending_verification_reviews,
    resolve_verification_session_review,
)
from app.models import ActivityItem, ReviewTask, VerificationSession
from app.schemas import (
    VerificationReviewDecisionInput,
    VerificationReviewQueueItemResponse,
    VerificationSessionStatusResponse,
)


class VerificationSessionReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def create_session_with_task(self, db, task_status="open"):
        session_id = uuid4()
        session = VerificationSession(
            id=str(session_id),
            employee_ref=f"EMP-{str(session_id)[:8]}",
            status="manual_review",
            match_score=0.61,
            liveness_score=0.88,
            quality_score=0.93,
            decision="review",
            algorithm_version="test-model",
            audit_ref=f"AUD-{str(session_id)[:8]}",
            is_match=True,
            is_live=False,
            details={"route": "manual_review", "review_tasks": []},
        )
        db.add(session)
        db.flush()
        task = ReviewTask(
            kind="verification",
            subject_ref=session.employee_ref,
            status=task_status,
            priority="high",
            summary="Нужна ручная проверка",
            verification_session_id=session.id,
        )
        db.add(task)
        db.commit()
        return session_id, session, task

    def test_queue_returns_only_open_and_in_review_verification_sessions(self):
        with self.session_factory() as db:
            open_id, _, _ = self.create_session_with_task(db, "open")
            in_review_id, _, _ = self.create_session_with_task(db, "in_review")
            self.create_session_with_task(db, "resolved")
            db.add(
                ReviewTask(
                    kind="health_id",
                    subject_ref="EMP-HEALTH",
                    status="open",
                    priority="medium",
                    summary="Проверка HEALTH_ID",
                )
            )
            db.commit()

            items = list_pending_verification_reviews(db)
            validated_items = [
                VerificationReviewQueueItemResponse.model_validate(item)
                for item in items
            ]

            self.assertEqual(len(validated_items), 2)
            self.assertEqual(
                {str(item.verification_session.id) for item in validated_items},
                {str(open_id), str(in_review_id)},
            )
            for item in items:
                self.assertEqual(set(item), {"verification_session"})
                self.assertEqual(
                    set(item["verification_session"]),
                    {
                        "id",
                        "employee_ref",
                        "status",
                        "match_score",
                        "liveness_score",
                        "quality_score",
                        "decision",
                    },
                )

    def test_session_review_updates_ground_truth_and_interface_side_effects_only(self):
        with self.session_factory() as db:
            session_id, session, task = self.create_session_with_task(db)

            resolved_task = resolve_verification_session_review(
                session_id,
                VerificationReviewDecisionInput(
                    status="reject",
                    ground_truth_match=False,
                    ground_truth_liveness=True,
                    comment="Ручная проверка: это другой человек.",
                ),
                db,
            )
            db.refresh(session)

            self.assertEqual(resolved_task.id, task.id)
            self.assertEqual(resolved_task.status, "resolved")
            self.assertEqual(resolved_task.resolution, "reject")
            self.assertEqual(
                resolved_task.reviewer_note,
                "Ручная проверка: это другой человек.",
            )
            self.assertEqual(session.status, "manual_review")
            self.assertEqual(session.decision, "review")
            self.assertTrue(session.is_match)
            self.assertFalse(session.is_live)
            self.assertFalse(session.ground_truth_match)
            self.assertTrue(session.ground_truth_liveness)
            self.assertEqual(session.details["review_tasks"][0]["status"], "resolved")
            self.assertEqual(
                db.scalar(select(func.count()).select_from(ActivityItem)),
                1,
            )

            status = get_verification_session_status(session_id, db)
            validated_status = VerificationSessionStatusResponse.model_validate(status)
            self.assertEqual(validated_status.status, "manual_review")
            self.assertEqual(validated_status.decision, "review")
            self.assertEqual(validated_status.review_status, "resolved")
            self.assertEqual(validated_status.review_resolution, "reject")
            self.assertFalse(validated_status.ground_truth_match)
            self.assertTrue(validated_status.ground_truth_liveness)

    def test_request_retry_does_not_overwrite_ground_truth_and_requires_active_task(self):
        with self.session_factory() as db:
            session_id, session, _ = self.create_session_with_task(db, "in_review")
            session.ground_truth_match = True
            session.ground_truth_liveness = False
            db.commit()

            resolve_verification_session_review(
                session_id,
                VerificationReviewDecisionInput(
                    status="request_retry",
                    ground_truth_match=False,
                    ground_truth_liveness=True,
                    comment="Запросить новый снимок.",
                ),
                db,
            )
            db.refresh(session)

            self.assertTrue(session.ground_truth_match)
            self.assertFalse(session.ground_truth_liveness)
            self.assertEqual(session.status, "manual_review")
            self.assertEqual(session.decision, "review")

            with self.assertRaises(HTTPException) as error:
                resolve_verification_session_review(
                    session_id,
                    VerificationReviewDecisionInput(status="approve"),
                    db,
                )
            self.assertEqual(error.exception.status_code, 409)

    def test_status_lookup_returns_not_found_for_unknown_session(self):
        with self.session_factory() as db:
            with self.assertRaises(HTTPException) as error:
                get_verification_session_status(uuid4(), db)
            self.assertEqual(error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
