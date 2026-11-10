import unittest

from app.biometrics import _resolve_verification_status, capture_quality_report
from app.verification_config import ModelConfig


class CameraCaptureQualityTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()

    def test_landscape_640_by_480_at_15_fps_meets_minimums(self):
        report = capture_quality_report(640, 480, 15, self.config)
        flags = report
        self.assertTrue(flags["resolution_ok"])
        self.assertTrue(flags["fps_ok"])
        self.assertEqual(report["resolution"]["actual"], [640, 480])
        self.assertEqual(report["resolution"]["required"], [640, 480])

    def test_portrait_480_by_640_does_not_meet_landscape_minimum(self):
        flags = capture_quality_report(480, 640, 30, self.config)
        self.assertFalse(flags["resolution_ok"])
        self.assertTrue(flags["fps_ok"])

    def test_small_frame_rate_metadata_drift_is_a_warning(self):
        report = capture_quality_report(640, 480, 14.92, self.config)
        self.assertTrue(report["resolution_ok"])
        self.assertFalse(report["fps_ok"])
        self.assertTrue(report["fps_acceptable"])
        self.assertEqual(report["fps"]["status"], "warning")
        self.assertEqual(report["fps"]["required"], 15)

    def test_frame_rate_more_than_one_percent_below_minimum_is_a_failure(self):
        report = capture_quality_report(640, 480, 14.8, self.config)
        self.assertTrue(report["resolution_ok"])
        self.assertFalse(report["fps_ok"])
        self.assertFalse(report["fps_acceptable"])
        self.assertEqual(report["fps"]["status"], "fail")

    def test_quality_score_above_threshold_passes_when_identity_and_liveness_pass(self):
        status = _resolve_verification_status(
            liveness_component_unavailable=False,
            blocking_issues=[],
            match_decision="match",
            match_score=0.901759,
            liveness_decision="live",
            quality_score=0.875,
            quality_threshold=self.config.capture_quality_threshold,
            manual_review_threshold=self.config.manual_review_threshold,
        )
        self.assertEqual(status, "verified")

    def test_quality_below_threshold_or_hard_block_still_needs_review(self):
        common = {
            "liveness_component_unavailable": False,
            "match_decision": "match",
            "match_score": 0.901759,
            "liveness_decision": "live",
            "quality_threshold": self.config.capture_quality_threshold,
            "manual_review_threshold": self.config.manual_review_threshold,
        }
        below_threshold = _resolve_verification_status(
            **common,
            blocking_issues=[],
            quality_score=0.849,
        )
        hard_block = _resolve_verification_status(
            **common,
            blocking_issues=["Разрешение видео ниже минимального."],
            quality_score=0.875,
        )
        self.assertEqual(below_threshold, "insufficient_quality")
        self.assertEqual(hard_block, "insufficient_quality")


if __name__ == "__main__":
    unittest.main()
