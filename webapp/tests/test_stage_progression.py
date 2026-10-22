"""按真实账号、HTTP 接口和 Jupyter 内核完成十阶段，并检查跨阶段门禁。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from webapp.curriculum import ALL_LESSONS, COURSES
from webapp.server import create_app


class StageProgressionTests(unittest.TestCase):
    def test_all_stages_unlock_only_after_the_previous_quiz(self):
        with TemporaryDirectory() as directory:
            app = create_app(Path(directory))
            with TestClient(app) as client:
                token = client.get("/api/auth/session").json()["token"]
                response = client.post("/api/auth/register", headers={"X-Lab-Token": token}, json={"username": "course_learner", "password": "Temporary-test-password-42"})
                self.assertEqual(response.status_code, 200, response.text)
                headers = {"X-Lab-Token": response.json()["token"]}
                total = 0
                for stage, course in COURSES.items():
                    state = client.get(f"/api/state?stage={stage}").json()
                    self.assertEqual(state["stage"], stage)
                    self.assertEqual(state["total"], len(course["lessons"]))
                    if stage < 10:
                        self.assertEqual(client.get(f"/api/state?stage={stage + 1}").status_code, 403)
                        next_id = COURSES[stage + 1]["lessons"][0]["id"]
                        self.assertEqual(client.get(f"/api/lessons/{next_id}").status_code, 403)
                        self.assertEqual(client.post("/api/execute", headers=headers, json={"lesson_id": next_id, "code": "print(1)"}).status_code, 403)
                        self.assertEqual(client.get(f"/api/export?stage={stage + 1}").status_code, 403)
                    for lesson in course["lessons"]:
                        visible = client.get(f"/api/lessons/{lesson['id']}")
                        self.assertEqual(visible.status_code, 200, visible.text)
                        self.assertNotIn("solution", visible.json())
                        self.assertNotIn("test", visible.json())
                        code = lesson["code"] if lesson["kind"] == "example" else lesson["solution"]
                        response = client.post("/api/execute", headers=headers, json={"lesson_id": lesson["id"], "code": code})
                        self.assertEqual(response.status_code, 200, (lesson["id"], response.text))
                        result = response.json()
                        self.assertTrue(result["passed"], (lesson["id"], result))
                        total += 1
                        self.assertEqual(result["state"]["overall_completed"], total)
                    # 只有完成代码还不能进入下一阶段。
                    if stage < 10:
                        self.assertEqual(client.get(f"/api/state?stage={stage + 1}").status_code, 403)
                        self.assertEqual(client.post("/api/quiz", headers=headers, json={"stage": stage + 1, "answers": COURSES[stage + 1]["answers"]}).status_code, 403)
                    wrong = [(value + 1) % 3 for value in course["answers"]]
                    self.assertFalse(client.post("/api/quiz", headers=headers, json={"stage": stage, "answers": wrong}).json()["passed"])
                    response = client.post("/api/quiz", headers=headers, json={"stage": stage, "answers": course["answers"]})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(response.json()["passed"])
                    self.assertTrue(response.json()["state"]["stage_complete"])
                    print(f"Stage {stage:02d}: {len(course['lessons'])} code cells and quiz passed", flush=True)
                final = client.get("/api/state").json()
                self.assertEqual(final["overall_completed"], len(ALL_LESSONS))
                self.assertTrue(final["course_complete"])
                self.assertTrue(all(item["status"] == "completed" for item in final["stages"]))
                self.assertEqual(client.get("/api/state?stage=1").json()["completed"], 21)
                self.assertEqual(client.get("/api/state?stage=11").status_code, 404)
                self.assertEqual(client.get("/api/materials/pretrained").status_code, 200)
