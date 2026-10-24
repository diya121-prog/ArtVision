import os
import json
import asyncio
import unittest
from PIL import Image
from io import BytesIO
from starlette.requests import Request
from fastapi.exceptions import HTTPException

from api.main import health, predict_endpoint, similar_endpoint


def make_request(method="GET", path="/", headers=None, query_string="", body=b""):
    headers = headers or {}
    raw_headers = [(k.lower().encode("latin-1"), str(v).encode("latin-1")) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw_headers,
        "query_string": query_string.encode("latin-1"),
    }
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    return Request(scope, receive)


class TestAPI(unittest.TestCase):
    def test_health_endpoint(self):
        data = health()
        self.assertEqual(data["status"], "ok")
        self.assertIn("device", data)
        self.assertIn("tasks_enabled", data)
        self.assertIn("style", data["tasks_enabled"])
        self.assertIn("authenticity", data["tasks_enabled"])
        self.assertEqual(data["total_catalog_artworks"], 8099)

    def test_predict_endpoint_contract(self):
        buf = BytesIO()
        im = Image.new("RGB", (224, 224), color=(180, 50, 50))
        im.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        req = make_request("POST", "/predict", headers={"Content-Type": "image/jpeg"}, body=image_bytes)
        res = asyncio.run(predict_endpoint(req, explain=False))
        self.assertEqual(res.status_code, 200)

        data = json.loads(res.body)
        self.assertEqual(data["schema_version"], "1.0")
        self.assertIn("style", data)
        self.assertIsNotNone(data["style"])
        self.assertIn("movement", data)
        self.assertIsNotNone(data["movement"])
        self.assertIn("authenticity", data)
        self.assertIn("caveat", data["authenticity"])
        self.assertIn("mood", data)
        self.assertTrue(data["mood"].get("experimental"))

    def test_predict_explain_endpoint(self):
        buf = BytesIO()
        im = Image.new("RGB", (224, 224), color=(60, 120, 200))
        im.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        req = make_request("POST", "/predict", headers={"Content-Type": "image/jpeg"}, body=image_bytes)
        res = asyncio.run(predict_endpoint(req, explain=True, explain_task="style"))
        self.assertEqual(res.status_code, 200)

        data = json.loads(res.body)
        self.assertIn("explanation", data)
        self.assertIsNotNone(data["explanation"])
        self.assertEqual(data["explanation"]["task"], "style")
        self.assertTrue(data["explanation"]["overlay_url"].startswith("/static/heatmaps/"))
        self.assertIn("border_mass_fraction", data["explanation"])

    def test_similar_endpoint(self):
        known_id = "ai_artbench_maxime-maufra_the-old-bridge-in-barbin-1884"
        payload = json.dumps({"image_id": known_id}).encode("utf-8")
        req = make_request("POST", "/similar", headers={"Content-Type": "application/json"}, body=payload)
        
        data = asyncio.run(similar_endpoint(req, top_k=5))
        self.assertEqual(data["schema_version"], "1.0")
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 5)

        retrieved_ids = [r["image_id"] for r in data["results"]]
        self.assertNotIn(known_id, retrieved_ids)
        for r in data["results"]:
            self.assertLessEqual(r["similarity"], 0.995)
            self.assertIn("image_url", r)

    def test_api_error_handling(self):
        # Empty body
        req_empty = make_request("POST", "/predict", headers={"Content-Type": "image/jpeg"}, body=b"")
        with self.assertRaises(HTTPException) as cm_empty:
            asyncio.run(predict_endpoint(req_empty))
        self.assertEqual(cm_empty.exception.status_code, 400)

        # Corrupt data
        req_corrupt = make_request("POST", "/predict", headers={"Content-Type": "image/jpeg"}, body=b"corrupt_non_image_data")
        with self.assertRaises(HTTPException) as cm_corrupt:
            asyncio.run(predict_endpoint(req_corrupt))
        self.assertEqual(cm_corrupt.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
