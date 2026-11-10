import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.arcface_service import ArcFaceIndexError, ArcFaceSearchService


class ArcFaceSearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)
        self.service = ArcFaceSearchService(self.storage_dir)
        self.embeddings = np.asarray([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        self.uuids = np.asarray([
            "employee-a",
            "employee-a",
            "employee-b",
            "employee-c",
        ])
        self.save_index_files()

    def tearDown(self):
        self.temp_dir.cleanup()

    def save_index_files(self):
        np.save(self.storage_dir / "employee_embeddings.npy", self.embeddings)
        np.save(self.storage_dir / "employee_uuids.npy", self.uuids)

    def test_loads_index_and_returns_distinct_employees_by_cosine(self):
        stats = self.service.load_index()
        result = self.service.search_embedding(
            np.asarray([0.8, 0.6], dtype=np.float32),
            top_k=3,
            threshold=0.7,
        )

        self.assertEqual(stats["vectors_count"], 4)
        self.assertEqual(stats["employees_count"], 3)
        self.assertEqual(result["found"], True)
        self.assertEqual([row["uuid"] for row in result["results"]], [
            "employee-a",
            "employee-b",
            "employee-c",
        ])
        self.assertAlmostEqual(
            result["results"][0]["cosine_similarity"], 1.0, places=5
        )
        self.assertAlmostEqual(
            result["results"][1]["cosine_similarity"], 0.6, places=5
        )
        self.assertEqual(result["top_match"]["uuid"], "employee-a")

    def test_returns_not_found_when_best_score_is_below_threshold(self):
        self.service.load_index()
        result = self.service.search_embedding(
            np.asarray([0.6, 0.8], dtype=np.float32),
            threshold=0.99,
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["top_match"]["uuid"], "employee-a")

    def test_compares_video_distribution_to_employee_reference_vectors(self):
        self.service.load_index()
        self.service.model = object()

        result = self.service.compare_video_embeddings(
            "employee-a",
            [
                np.asarray([0.8, 0.6], dtype=np.float32),
                np.asarray([0.6, 0.8], dtype=np.float32),
            ],
        )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["reference_vectors"], 2)
        self.assertEqual(result["video_vectors"], 2)
        self.assertIsNotNone(result["mean"]["centroid_cosine_similarity"])
        self.assertIsNotNone(result["variance"]["ratio"])
        self.assertIsNone(result["outliers"]["count"])
        self.assertEqual(
            len(result["distribution"]["bin_centers"]),
            len(result["distribution"]["reference_counts"]),
        )
        self.assertEqual(
            len(result["distribution"]["bin_centers"]),
            len(result["distribution"]["video_counts"]),
        )

    def test_marks_missing_employee_reference_without_inventing_statistics(self):
        self.service.load_index()
        self.service.model = object()

        result = self.service.compare_video_embeddings(
            "employee-missing",
            [np.asarray([0.8, 0.6], dtype=np.float32)],
        )

        self.assertEqual(result["status"], "no_reference")
        self.assertEqual(result["reference_vectors"], 0)
        self.assertIsNone(result["variance"]["ratio"])
        self.assertIsNone(result["outliers"]["count"])

    def test_normalizes_unscaled_vectors_before_indexing(self):
        self.embeddings = self.embeddings * 3
        self.save_index_files()

        self.service.load_index()
        result = self.service.search_embedding(
            np.asarray([0.8, 0.6], dtype=np.float32),
            threshold=0.99,
        )

        self.assertTrue(result["found"])
        self.assertAlmostEqual(
            result["top_match"]["cosine_similarity"], 1.0, places=5
        )

    def test_missing_index_files_raise_a_clear_file_error(self):
        (self.storage_dir / "employee_embeddings.npy").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "employee_embeddings.npy"):
            self.service.load_index()

    def test_vector_and_uuid_row_mismatch_is_rejected(self):
        np.save(self.storage_dir / "employee_uuids.npy", self.uuids[:2])

        with self.assertRaisesRegex(ArcFaceIndexError, "не совпадает"):
            self.service.load_index()

    def test_zero_vectors_are_rejected(self):
        self.embeddings[0] = [0.0, 0.0]
        self.save_index_files()

        with self.assertRaisesRegex(ArcFaceIndexError, "нулевой эмбеддинг"):
            self.service.load_index()

    def test_enrollment_persists_alongside_the_training_index(self):
        self.service.load_index()

        result = self.service.add_embedding(
            "employee-d",
            np.asarray([0.6, 0.8], dtype=np.float32),
        )

        self.assertTrue(result["added"])
        self.assertFalse(result["updated"])
        self.assertEqual(result["vectors_count"], 5)
        np.testing.assert_array_equal(
            np.load(self.storage_dir / "employee_embeddings.npy"),
            self.embeddings,
        )
        self.assertTrue(
            (self.storage_dir / "employee_enrolled_embeddings.npy").is_file()
        )

        reloaded_service = ArcFaceSearchService(self.storage_dir)
        stats = reloaded_service.load_index()
        self.assertEqual(stats["vectors_count"], 5)
        self.assertEqual(stats["employees_count"], 4)
        result = reloaded_service.search_embedding(
            np.asarray([0.6, 0.8], dtype=np.float32),
        )
        self.assertEqual(result["top_match"]["uuid"], "employee-d")

    def test_repeated_enrollment_of_the_same_embedding_is_idempotent(self):
        self.service.load_index()
        embedding = np.asarray([0.6, 0.8], dtype=np.float32)

        first = self.service.add_embedding("employee-d", embedding)
        second = self.service.add_embedding("employee-d", embedding)

        self.assertTrue(first["added"])
        self.assertFalse(second["added"])
        self.assertTrue(second["already_indexed"])
        self.assertEqual(second["vectors_count"], 5)
        self.assertEqual(
            len(np.load(self.storage_dir / "employee_enrolled_uuids.npy")),
            1,
        )

    def test_adds_multiple_video_vectors_and_persists_them_across_reload(self):
        self.service.load_index()
        video_vectors = [
            np.asarray([0.6, 0.8], dtype=np.float32),
            np.asarray([0.8, 0.6], dtype=np.float32),
        ]

        result = self.service.add_video_embeddings("employee-d", video_vectors)

        self.assertEqual(result["added_vectors"], 2)
        self.assertEqual(result["video_vectors_count"], 2)
        saved_uuids = np.load(
            self.storage_dir / "employee_video_uuids.npy",
            allow_pickle=False,
        )
        self.assertEqual(saved_uuids.tolist(), ["employee-d", "employee-d"])

        duplicate_result = self.service.add_video_embeddings(
            "employee-d", video_vectors
        )
        self.assertEqual(duplicate_result["added_vectors"], 0)
        self.assertEqual(duplicate_result["skipped_vectors"], 2)

        reloaded_service = ArcFaceSearchService(self.storage_dir)
        stats = reloaded_service.load_index()
        self.assertEqual(stats["vectors_count"], 6)
        self.assertEqual(reloaded_service.status()["video_vectors_count"], 2)
        self.assertEqual(
            reloaded_service.search_embedding(video_vectors[0])["top_match"]["uuid"],
            "employee-d",
        )

    def test_manual_enrollment_keeps_video_vectors_in_the_index(self):
        self.service.load_index()
        self.service.add_video_embeddings(
            "employee-d",
            [np.asarray([0.6, 0.8], dtype=np.float32)],
        )

        self.service.add_embedding(
            "employee-d",
            np.asarray([-0.6, 0.8], dtype=np.float32),
        )

        reloaded_service = ArcFaceSearchService(self.storage_dir)
        stats = reloaded_service.load_index()
        self.assertEqual(stats["vectors_count"], 6)
        self.assertEqual(reloaded_service.status()["video_vectors_count"], 1)

    def test_new_reference_replaces_only_its_previous_manual_embedding(self):
        self.service.load_index()

        first = self.service.add_embedding(
            "employee-a",
            np.asarray([-0.6, 0.8], dtype=np.float32),
        )
        second = self.service.add_embedding(
            "employee-a",
            np.asarray([0.6, -0.8], dtype=np.float32),
        )

        self.assertTrue(first["added"])
        self.assertTrue(second["updated"])
        self.assertEqual(second["vectors_count"], 5)
        enrolled_uuids = np.load(
            self.storage_dir / "employee_enrolled_uuids.npy"
        )
        self.assertEqual(enrolled_uuids.tolist(), ["employee-a"])


if __name__ == "__main__":
    unittest.main()
