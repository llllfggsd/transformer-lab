"""本地账号与服务端会话。密码和会话凭据均不以明文写入数据库。"""

from contextlib import contextmanager
import hashlib
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
import uuid

from fastapi import HTTPException

COOKIE_NAME = "transformer_lab_session"
SESSION_SECONDS = 7 * 24 * 60 * 60


def password_hash(password, salt=None):
    salt = secrets.token_hex(16) if salt is None else salt
    digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=2**17, r=8, p=1, maxmem=256 * 1024 * 1024).hex()
    return f"scrypt$131072$8$1${salt}${digest}"


def password_matches(password, stored):
    try:
        algorithm, n, r, p, salt, digest = stored.split("$")
        if (algorithm, n, r, p) != ("scrypt", "131072", "8", "1"):
            return False
        return secrets.compare_digest(password_hash(password, salt), stored)
    except (ValueError, TypeError):
        return False


class Accounts:
    def __init__(self, root):
        self.root = root
        self.database = root / "accounts.sqlite3"
        self.bootstrap_token = secrets.token_urlsafe(32)
        self.registration_lock = threading.Lock()
        self.attempts_lock = threading.Lock()
        self.failed_attempts = {}
        self.dummy_hash = password_hash(secrets.token_urlsafe(24))
        with self.connect() as database:
            database.execute("CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT NOT NULL, username_key TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, created_at INTEGER NOT NULL)")
            database.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), csrf_token TEXT NOT NULL, expires_at INTEGER NOT NULL)")
            database.execute("CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def legacy_available(self):
        with self.connect() as database:
            if database.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                return False
        old = self.root / "progress.sqlite3"
        if not old.exists():
            return False
        connection = sqlite3.connect(old)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            return "lessons" in tables and connection.execute("SELECT 1 FROM lessons LIMIT 1").fetchone() is not None
        finally:
            connection.close()

    def first_account(self):
        with self.connect() as database:
            return database.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None

    def register(self, username, password, import_legacy=False):
        username = unicodedata.normalize("NFKC", username).strip()
        if not re.fullmatch(r"[\w.-]{3,32}", username):
            raise HTTPException(422, "用户名需为 3–32 位，可使用中文、字母、数字、下划线、点或短横线。")
        if not 8 <= len(password) <= 128:
            raise HTTPException(422, "密码长度需为 8–128 位。")
        hashed = password_hash(password)
        user_id = uuid.uuid4().hex
        imported = False
        claimed = False
        with self.registration_lock:
            try:
                with self.connect() as database:
                    database.execute("BEGIN IMMEDIATE")
                    first = database.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None
                    database.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (user_id, username, username.casefold(), hashed, int(time.time())))
                    claimed = first and import_legacy
                    if claimed and (self.root / "progress.sqlite3").exists():
                        directory = self.root / "users" / user_id
                        directory.mkdir(parents=True, exist_ok=True)
                        source = sqlite3.connect(self.root / "progress.sqlite3")
                        target = sqlite3.connect(directory / "progress.sqlite3")
                        try:
                            source.backup(target)
                            imported = True
                        finally:
                            source.close()
                            target.close()
            except sqlite3.IntegrityError as error:
                raise HTTPException(409, "这个用户名已被注册，请更换用户名或直接登录。") from error
        return {"id": user_id, "username": username}, claimed, imported

    def login(self, username, password, client_key):
        key = unicodedata.normalize("NFKC", username).strip().casefold()
        attempt_key = (client_key, key)
        now = time.time()
        with self.attempts_lock:
            # 只保留近期失败记录，防止不同用户名让内存不断增长。
            self.failed_attempts = {k: v for k, v in self.failed_attempts.items() if v and v[-1] > now - 900}
            failures = [stamp for stamp in self.failed_attempts.get(attempt_key, []) if stamp > now - 900]
            if len(failures) >= 10:
                raise HTTPException(429, "登录失败次数较多，请 15 分钟后再试。")
        with self.connect() as database:
            user = database.execute("SELECT * FROM users WHERE username_key=?", (key,)).fetchone()
        correct = password_matches(password, user["password_hash"] if user else self.dummy_hash)
        if not user or not correct:
            with self.attempts_lock:
                self.failed_attempts.setdefault(attempt_key, []).append(now)
            raise HTTPException(401, "用户名或密码不正确。")
        with self.attempts_lock:
            self.failed_attempts.pop(attempt_key, None)
        return {"id": user["id"], "username": user["username"]}

    def new_session(self, user):
        token = secrets.token_urlsafe(48)
        csrf = secrets.token_urlsafe(32)
        with self.connect() as database:
            database.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
            database.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (hashlib.sha256(token.encode()).hexdigest(), user["id"], csrf, int(time.time()) + SESSION_SECONDS))
        return token, csrf

    def session(self, token):
        if not token or len(token) > 256:
            return None
        with self.connect() as database:
            row = database.execute("SELECT users.id, users.username, sessions.csrf_token FROM sessions JOIN users ON users.id=sessions.user_id WHERE token_hash=? AND expires_at>?", (hashlib.sha256(token.encode()).hexdigest(), int(time.time()))).fetchone()
        if not row:
            return None
        return {"user": {"id": row["id"], "username": row["username"]}, "token": row["csrf_token"]}

    def logout(self, token):
        with self.connect() as database:
            database.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def has_sessions(self, user_id):
        with self.connect() as database:
            return database.execute("SELECT 1 FROM sessions WHERE user_id=? AND expires_at>? LIMIT 1", (user_id, int(time.time()))).fetchone() is not None
