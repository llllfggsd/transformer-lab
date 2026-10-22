"""关键学习流程的集成检查；真实执行课程，进度只写入临时目录。"""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from webapp.curriculum import LESSONS, QUIZ_ANSWERS, STAGE_DIR
from webapp.server import create_app


class LearningLabTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.app = create_app(Path(self.directory.name))
        self.client = TestClient(self.app)
        self.client.__enter__()
        bootstrap = self.client.get("/api/auth/session").json()["token"]
        response = self.client.post("/api/auth/register", json={"username": "learner", "password": "Test-only-password-42"}, headers={"X-Lab-Token": bootstrap})
        self.assertEqual(response.status_code, 200, response.text)
        self.user_id = response.json()["user"]["id"]
        self.headers = {"X-Lab-Token": self.client.get("/api/state").json()["token"]}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def post(self, path, payload=None):
        return self.client.post(path, json=payload, headers=self.headers)

    def run_lesson(self, index, code):
        response = self.post("/api/execute", {"lesson_id": LESSONS[index]["id"], "code": code})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_guards_and_drafts(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/api/lessons/s1-example-02").status_code, 403)
        self.assertEqual(self.post("/api/draft", {"lesson_id": "s1-example-02", "code": "print(1)"}).status_code, 403)
        self.assertEqual(self.post("/api/execute", {"lesson_id": "s1-example-02", "code": "print(1)"}).status_code, 403)
        self.assertEqual(self.client.get("/api/quiz").status_code, 403)
        self.assertEqual(self.post("/api/quiz", {"answers": QUIZ_ANSWERS}).status_code, 403)
        payload = {"lesson_id": "s1-example-01", "code": "print('草稿')"}
        self.assertEqual(self.client.post("/api/draft", json=payload).status_code, 403)
        self.assertEqual(self.client.post("/api/draft", json=payload, headers={**self.headers, "Origin": "https://other.example"}).status_code, 403)
        self.assertEqual(self.post("/api/draft", payload).status_code, 200)
        self.assertEqual(self.client.get("/api/lessons/s1-example-01").json()["draft"], payload["code"])
        self.assertEqual(self.client.get("/api/state").json()["completed"], 0)
        exported = json.loads(self.client.get("/api/export").text)
        self.assertEqual(len(exported["cells"]), 2)
        self.assertEqual(exported["cells"][1]["source"], payload["code"])

    def test_full_curriculum_and_restart(self):
        result = self.run_lesson(0, "print('可以运行，但不是示例')")
        self.assertTrue(result["ok"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["state"]["frontier"], 0)
        result = self.run_lesson(0, "if :")
        self.assertFalse(result["ok"])
        self.assertFalse(result["passed"])
        screenshot_code = LESSONS[0]["code"].replace(" = ", "=").replace("，今天计划学习 ", ",今天计划学习").replace("} 分钟", "}分钟")
        accepted = self.run_lesson(0, screenshot_code)
        self.assertTrue(accepted["ok"])
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["state"]["completed"], 1)
        self.assertEqual(accepted["differences"], [])
        mismatch = self.run_lesson(0, screenshot_code.replace("学习分钟数", "学习秒数"))
        self.assertTrue(mismatch["ok"])
        self.assertFalse(mismatch["passed"])
        self.assertEqual(mismatch["differences"][0]["line"], 7)
        source = (STAGE_DIR / "04_reference_answers.py").read_text(encoding="utf-8")
        solutions = [ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        for index, lesson in enumerate(LESSONS):
            if index == 2:
                self.assertEqual(self.post("/api/kernel/restart").status_code, 200)
            if index == 13:
                # 定义本身可执行，边界用例不满足时也不能获得通过记录。
                bad = self.run_lesson(index, "def score_label(score):\n    return '通过'")
                self.assertTrue(bad["ok"])
                self.assertFalse(bad["passed"])
                self.assertEqual(bad["state"]["frontier"], index)
            code = lesson["code"] if lesson["kind"] == "example" else solutions[index - 13]
            result = self.run_lesson(index, code)
            self.assertTrue(result["ok"], (lesson["id"], result))
            self.assertTrue(result["passed"], (lesson["id"], result))
            self.assertEqual(result["state"]["completed"], index + 1)
        state = self.client.get("/api/state").json()
        self.assertFalse(state["stage_complete"])
        self.assertTrue(state["quiz_available"])
        wrong = self.post("/api/quiz", {"answers": [0, 0, 0]}).json()
        self.assertFalse(wrong["passed"])
        result = self.post("/api/quiz", {"answers": QUIZ_ANSWERS}).json()
        self.assertTrue(result["passed"])
        self.assertTrue(result["state"]["stage_complete"])
        self.assertEqual(result["state"]["stages"][1]["status"], "available")
        # 同一个数据目录重新启动应用，成绩保留，内核重新建立。
        self.client.__exit__(None, None, None)
        cookies = dict(self.client.cookies)
        self.app = create_app(Path(self.directory.name))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.client.cookies.update(cookies)
        restored = self.client.get("/api/state?stage=1").json()
        self.assertEqual(restored["completed"], 21)
        self.assertTrue(restored["stage_complete"])
        self.assertFalse(restored["kernel_ready"])
        self.assertEqual(self.client.get("/api/state").json()["stage"], 2)

    def test_timeout_and_recovery(self):
        lab = self.app.state.labs.get(self.user_id)
        def timeout():
            lab.kernel.start()
            return lab.kernel.execute("while True:\n    pass", timeout=1)
        result = lab.worker.submit(timeout).result(timeout=40)
        self.assertFalse(result["ok"])
        self.assertIsNone(lab.kernel.client)
        self.assertIn("超过", result["outputs"][0]["text"])
        result = self.run_lesson(0, LESSONS[0]["code"])
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
