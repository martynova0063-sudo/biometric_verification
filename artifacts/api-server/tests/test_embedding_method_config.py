import unittest

import numpy as np
from fastapi import HTTPException

from app.biometrics import _embedding
from app.verification_config import PARAMETER_OPTION_VALUES, parameter_schema


class EmbeddingMethodConfigTests(unittest.TestCase):
    def test_resnet50_arcface_is_an_editable_embedding_method(self):
        self.assertIn(
            "resnet50_arcface",
            PARAMETER_OPTION_VALUES["embedding_method"],
        )
        field = next(
            item
            for item in parameter_schema()
            if item["key"] == "embedding_method"
        )
        self.assertIn(
            "resnet50_arcface",
            {option["value"] for option in field["options"]},
        )

    def test_uses_and_normalizes_the_trained_model_embedding_callback(self):
        face = np.zeros((16, 16, 3), dtype=np.uint8)
        result = _embedding(
            face,
            "resnet50_arcface",
            model_embedding_callback=lambda _crop: np.asarray(
                [3.0, 4.0], dtype=np.float32
            ),
        )

        np.testing.assert_allclose(result, [0.6, 0.8])

    def test_reports_unavailable_trained_model_instead_of_falling_back(self):
        face = np.zeros((16, 16, 3), dtype=np.uint8)

        with self.assertRaises(HTTPException) as raised:
            _embedding(face, "resnet50_arcface")

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
