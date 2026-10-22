"""注册会话、账号隔离、原进度迁移和真实内核隔离检查。"""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from webapp.accounts import COOKIE_NAME
from webapp.curriculum import LESSONS, QUIZ_ANSWERS
from webapp.server import Lab, create_app

PASSWORD = "Test-password-only-42"


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.app = create_app(self.root)
        self.host = TestClient(self.app)
        self.host.__enter__()
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.host.__exit__(None, None, None)
        self.directory.cleanup()

    def client(self):
        client = TestClient(self.app)
        self.clients.append(client)
        return client

    def register(self, username, import_legacy=False):
        client = self.client()
        token = client.get("/api/auth/session").json()["token"]
        response = client.post("/api/auth/register", headers={"X-Lab-Token": token}, json={"username": username, "password": PASSWORD, "import_legacy": import_legacy})
        self.assertEqual(response.status_code, 200, response.text)
        return client, response.json()

    @staticmethod
    def headers(session):
        return {"X-Lab-Token": session["token"], "X-Lab-User": session["user"]["id"]}

    def test_authentication_and_sessions(self):
        for path in ["/api/state", "/api/lessons/s1-example-01", "/api/export", "/api/quiz"]:
            self.assertEqual(self.host.get(path).status_code, 401)
        payload = {"username": "Alice", "password": PASSWORD}
        self.assertEqual(self.host.post("/api/auth/register", json=payload).status_code, 403)
        self.assertEqual(self.host.post("/api/auth/register", json=payload, headers={b"X-Lab-Token": b"\xe9"}).status_code, 403)
        bootstrap = self.host.get("/api/auth/session").json()["token"]
        self.assertEqual(self.host.post("/api/auth/register", json=payload, headers={"X-Lab-Token": bootstrap, "Origin": "https://other.example"}).status_code, 403)
        self.assertEqual(self.host.post("/api/auth/register", json={**payload, "password": "short"}, headers={"X-Lab-Token": bootstrap}).status_code, 422)
        alice, session = self.register("Alice")
        raw_cookie = alice.cookies.get(COOKIE_NAME)
        self.assertTrue(raw_cookie)
        self.assertTrue(alice.get("/api/auth/session").json()["authenticated"])
        with self.app.state.accounts.connect() as db:
            stored = db.execute("SELECT password_hash FROM users").fetchone()[0]
            cookie_hash = db.execute("SELECT token_hash FROM sessions").fetchone()[0]
        self.assertNotIn(PASSWORD, stored)
        self.assertTrue(stored.startswith("scrypt$"))
        self.assertNotEqual(raw_cookie, cookie_hash)
        self.assertEqual(self.host.post("/api/auth/register", json={**payload, "username": "alice"}, headers={"X-Lab-Token": bootstrap}).status_code, 409)
        invalid = self.host.post("/api/auth/login", json={**payload, "password": "incorrect-password"}, headers={"X-Lab-Token": bootstrap})
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(invalid.json()["detail"], "用户名或密码不正确。")
        logged_in = self.host.post("/api/auth/login", json={**payload, "username": "ALICE"}, headers={"X-Lab-Token": bootstrap})
        self.assertEqual(logged_in.status_code, 200)
        cookies = logged_in.headers["set-cookie"].lower()
        self.assertIn("httponly", cookies)
        self.assertIn("samesite=strict", cookies)
        self.assertNotEqual(raw_cookie, self.host.cookies.get(COOKIE_NAME))
        self.assertEqual(alice.post("/api/auth/logout", headers=self.headers(session)).status_code, 200)
        self.assertEqual(alice.get("/api/state").status_code, 401)
        # 一个浏览器退出不会撤销该用户其他浏览器的独立登录会话。
        self.assertEqual(self.host.get("/api/state").status_code, 200)
        with self.app.state.accounts.connect() as db:
            db.execute("UPDATE sessions SET expires_at=1")
        self.assertEqual(self.host.get("/api/state").status_code, 401)

    def test_drafts_progress_export_quiz_and_kernels_are_separate(self):
        alice, a = self.register("Alice")
        bob, b = self.register("Bob")
        ah, bh = self.headers(a), self.headers(b)
        response = alice.post("/api/execute", headers=ah, json={"lesson_id": "s1-example-01", "code": LESSONS[0]["code"]})
        self.assertTrue(response.json()["passed"], response.text)
        self.assertEqual(alice.get("/api/state").json()["completed"], 1)
        self.assertEqual(bob.get("/api/state").json()["completed"], 0)
        self.assertEqual(bob.get("/api/lessons/s1-example-02").status_code, 403)
        for client, headers, text in [(alice, ah, "alice-private-draft"), (bob, bh, "bob-private-draft")]:
            response = client.post("/api/draft", headers=headers, json={"lesson_id": "s1-example-01", "code": text, "user_id": a["user"]["id"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get("/api/lessons/s1-example-01").json()["draft"], text)
            self.assertIn(text, client.get("/api/export").text)
        self.assertNotIn("alice-private-draft", bob.get("/api/export").text)
        stale = bob.get("/api/state", headers=ah)
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["code"], "ACCOUNT_CHANGED")
        self.assertEqual(bob.post("/api/draft", headers={"X-Lab-Token": a["token"]}, json={"lesson_id": "s1-example-01", "code": "stale-account-draft"}).status_code, 403)
        alice.post("/api/execute", headers=ah, json={"lesson_id": "s1-example-01", "code": "private_value = 'alice'"})
        result = bob.post("/api/execute", headers=bh, json={"lesson_id": "s1-example-01", "code": "print(globals().get('private_value', 'missing'))"}).json()
        self.assertTrue(result["ok"])
        self.assertEqual(result["outputs"][0]["text"].strip(), "missing")
        al = self.app.state.labs.get(a["user"]["id"])
        bl = self.app.state.labs.get(b["user"]["id"])
        self.assertNotEqual(al.data_dir, bl.data_dir)
        self.assertIsNot(al.kernel.manager, bl.kernel.manager)
        with al.connect() as db:
            for lesson in LESSONS:
                db.execute("INSERT INTO lessons(id, code) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET code=excluded.code", (lesson["id"], lesson["code"]))
        self.assertTrue(alice.post("/api/quiz", headers=ah, json={"answers": QUIZ_ANSWERS}).json()["passed"])
        self.assertTrue(alice.get("/api/state?stage=1").json()["stage_complete"])
        self.assertFalse(bob.get("/api/state").json()["stage_complete"])
        self.assertEqual(bob.get("/api/quiz").status_code, 403)
        self.assertEqual(bob.post("/api/auth/logout", headers=bh).status_code, 200)
        self.assertIsNone(bl.kernel.client)
        self.assertIsNotNone(al.kernel.client)
        # 再次登录后，保留该用户自己的草稿与进度。
        bootstrap = bob.get("/api/auth/session").json()["token"]
        response = bob.post("/api/auth/login", headers={"X-Lab-Token": bootstrap}, json={"username": "Bob", "password": PASSWORD})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(bob.get("/api/state").json()["completed"], 0)

    def test_legacy_progress_is_imported_once_and_preserved(self):
        legacy = Lab(self.root)
        try:
            legacy.save_draft("s1-example-01", "old-personal-draft")
            with legacy.connect() as db:
                db.execute("UPDATE lessons SET code=? WHERE id=?", (LESSONS[0]["code"], "s1-example-01"))
        finally:
            legacy.close()
        self.assertTrue(self.host.get("/api/auth/session").json()["legacy_available"])
        alice, response = self.register("Alice", import_legacy=True)
        self.assertTrue(response["legacy_claimed"])
        self.assertTrue(response["legacy_imported"])
        self.assertEqual(alice.get("/api/state").json()["completed"], 1)
        self.assertEqual(alice.get("/api/lessons/s1-example-01").json()["draft"], "old-personal-draft")
        bob, result = self.register("Bob", import_legacy=True)
        self.assertFalse(result["legacy_claimed"])
        self.assertEqual(bob.get("/api/state").json()["completed"], 0)
        connection = sqlite3.connect(self.root / "progress.sqlite3")
        try:
            self.assertEqual(connection.execute("SELECT draft FROM lessons").fetchone()[0], "old-personal-draft")
        finally:
            connection.close()

    def test_login_failure_limit(self):
        token = self.host.get("/api/auth/session").json()["token"]
        with patch("webapp.accounts.password_matches", return_value=False):
            for _ in range(10):
                response = self.host.post("/api/auth/login", headers={"X-Lab-Token": token}, json={"username": "unknown", "password": PASSWORD})
                self.assertEqual(response.status_code, 401)
            self.assertEqual(self.host.post("/api/auth/login", headers={"X-Lab-Token": token}, json={"username": "unknown", "password": PASSWORD}).status_code, 429)


if __name__ == "__main__":
    unittest.main()
