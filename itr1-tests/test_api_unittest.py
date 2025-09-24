import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

class TestAlphabetMasteryAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from main import app
        cls.client = TestClient(app)

    @patch("main.detect_handwritten_letters_from_base64", autospec=True)
    def test_success(self, mock_detect):
        mock_detect.return_value = {
            "expected_letter": "A",
            "is_correct": True,
            "reason": "match found with sufficient confidence",
            "detected_count": 3,
            "match_count": 2,
            "top_match_confidence": 0.92,
            "match_ratio": 0.667,
            "letters": [{"letter": "A", "confidence": 0.9}],
            "mismatches": [{"letter": "B", "count": 1, "top_confidence": 0.4}],
        }

        payload = {"canvas_input": "base64here", "expected_letter": "A"}
        resp = self.client.post("/alphabet_mastery", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["is_correct"])
        self.assertEqual(data["expected_letter"], "A")

    @patch("main.detect_handwritten_letters_from_base64", autospec=True)
    def test_detector_http_exception_bubbles(self, mock_detect):
        from fastapi import HTTPException
        mock_detect.side_effect = HTTPException(status_code=400, detail="expected_letter must be a single A–Z letter")

        payload = {"canvas_input": "base64here", "expected_letter": "AB"}
        resp = self.client.post("/alphabet_mastery", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("expected_letter", resp.text)

    @patch("main.detect_handwritten_letters_from_base64", autospec=True)
    def test_detector_generic_error_becomes_500(self, mock_detect):
        mock_detect.side_effect = RuntimeError("boom")

        payload = {"canvas_input": "base64here", "expected_letter": "A"}
        resp = self.client.post("/alphabet_mastery", json=payload)
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Unexpected error", resp.text)

    def test_missing_api_key_env(self):
        
        # Remove env to trigger 500 from main.read_canvas_input
        with self.subTest("no GCV_API_KEY"):
            from os import environ
            from importlib import reload
            import main as main_module

            # Temporarily clear env and reload main to re-evaluate anything if needed
            old = environ.pop("GCV_API_KEY", None)
            try:
                
                # Test via client (FastAPI code checks env at request time, not import time)
                client = TestClient(main_module.app)
                resp = client.post("/alphabet_mastery", json={"canvas_input": "x", "expected_letter": "A"})
                self.assertEqual(resp.status_code, 500)
                self.assertIn("Missing GCV_API_KEY", resp.text)
            finally:
                if old is not None:
                    environ["GCV_API_KEY"] = old