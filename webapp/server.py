"""仅监听本机的学习服务，课程进度与阶段门禁在服务端维护。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
import json
import importlib.util
from pathlib import Path
import secrets
import sqlite3
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .curriculum import ALL_LESSONS, COURSES, LESSON_LOOKUP, ROOT
from .accounts import Accounts, COOKIE_NAME, SESSION_SECONDS
from .kernel import PythonKernel
from .validation import compare_example

STATIC = Path(__file__).with_name("static")


class CodeRequest(BaseModel):
    lesson_id: str
    code: str = Field(max_length=40000)


class QuizRequest(BaseModel):
    answers: list[int] = Field(min_length=3, max_length=3)
    stage: int = Field(default=1, ge=1, le=10)


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    import_legacy: bool = False


class Lab:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database = self.data_dir / "progress.sqlite3"
        self.lock = threading.Lock()
        # ZMQ socket 始终由同一个线程创建与使用。
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jupyter-kernel")
        self.kernel = PythonKernel(self.data_dir)
        with self.connect() as database:
            database.execute("CREATE TABLE IF NOT EXISTS lessons (id TEXT PRIMARY KEY, draft TEXT NOT NULL DEFAULT '', code TEXT, outputs TEXT NOT NULL DEFAULT '[]')")
            database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def records(self):
        with self.connect() as database:
            return {row["id"]: dict(row) for row in database.execute("SELECT * FROM lessons")}

    def completed(self, stage=1):
        with self.connect() as database:
            row = database.execute("SELECT value FROM metadata WHERE key=?", (f"stage_{stage}_complete",)).fetchone()
        return bool(row and row["value"] == "true")

    def stage_unlocked(self, stage):
        return stage in COURSES and all(self.completed(previous) for previous in range(1, stage))

    def require_stage(self, stage):
        if stage not in COURSES:
            raise HTTPException(404, "课程阶段不存在。")
        if not self.stage_unlocked(stage):
            raise HTTPException(403, "请先完成上一阶段的示例、练习和阶段验收，不能跳阶段。")

    def resume_stage(self):
        return next((stage for stage in COURSES if not self.completed(stage)), max(COURSES))

    def frontier(self, records=None, stage=1):
        records = self.records() if records is None else records
        lessons = COURSES[stage]["lessons"]
        return next((index for index, lesson in enumerate(lessons) if not records.get(lesson["id"], {}).get("code")), len(lessons))

    def lesson_location(self, lesson_id):
        if lesson_id not in LESSON_LOOKUP:
            raise HTTPException(404, "课程代码段不存在。")
        stage, index = LESSON_LOOKUP[lesson_id]
        self.require_stage(stage)
        if index > self.frontier(stage=stage):
            raise HTTPException(403, "请先完成前面的代码段，不能跳段学习。")
        return stage, index

    def state(self, stage=None):
        stage = self.resume_stage() if stage is None else stage
        self.require_stage(stage)
        course = COURSES[stage]
        lessons = course["lessons"]
        records = self.records()
        frontier = self.frontier(records, stage)
        complete = self.completed(stage)
        finished = {number: self.completed(number) for number in COURSES}
        return {
            "stage": stage, "stage_title": course["title"], "stage_description": course["description"], "stage_tag": course["tag"],
            "kernel_ready": self.kernel.client is not None,
            "frontier": frontier, "total": len(lessons), "completed": frontier,
            "stage_complete": complete, "quiz_available": frontier == len(lessons), "quiz_count": len(course["quiz"]),
            "next_stage": stage + 1 if complete and stage < max(COURSES) else None,
            "course_complete": all(finished.values()), "overall_total": len(ALL_LESSONS),
            "overall_completed": sum(bool(records.get(item["id"], {}).get("code")) for item in ALL_LESSONS),
            "requirements": course["requirements"],
            "missing_packages": [name for name in course["requirements"] if importlib.util.find_spec(name) is None],
            "sources": course["sources"],
            "stages": [{"number": number, "title": item["title"], "available": all(finished[n] for n in range(1, number)),
                        "completed": self.frontier(records, number), "total": len(item["lessons"]),
                        "status": "completed" if finished[number] else "locked" if not all(finished[n] for n in range(1, number)) else "current" if number == stage else "available"}
                       for number, item in COURSES.items()],
            "lessons": [{"id": lesson["id"], "title": lesson["title"], "kind": lesson["kind"],
                         "completed": bool(records.get(lesson["id"], {}).get("code")), "unlocked": index <= frontier}
                        for index, lesson in enumerate(lessons)],
        }

    def lesson(self, lesson_id):
        stage, index = self.lesson_location(lesson_id)
        lesson = {key: value for key, value in COURSES[stage]["lessons"][index].items() if key not in ("test", "solution")}
        record = self.records().get(lesson_id, {})
        lesson.update({"index": index, "draft": record.get("draft", ""), "outputs": json.loads(record.get("outputs", "[]")), "completed": bool(record.get("code"))})
        return lesson

    def save_draft(self, lesson_id, code):
        self.lesson_location(lesson_id)
        with self.connect() as database:
            database.execute("INSERT INTO lessons(id, draft) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET draft=excluded.draft", (lesson_id, code))

    def prepare_kernel(self, stage, index):
        if self.kernel.client and self.kernel.position == (stage, index):
            return
        self.kernel.start()
        records = self.records()
        for previous in COURSES[stage]["lessons"][:index]:
            result = self.kernel.execute(records[previous["id"]]["code"], timeout=previous.get("timeout", 20))
            if not result["ok"]:
                self.kernel.close()
                raise HTTPException(409, "恢复先前的变量状态失败，请返回上一段重新运行。")
        self.kernel.position = (stage, index)

    def execute(self, lesson_id, code):
        if not code.strip():
            raise HTTPException(422, "请先在右侧手敲代码。")
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "内核正在运行，请等待当前代码完成。")
        try:
            return self.worker.submit(self._execute, lesson_id, code).result()
        finally:
            self.lock.release()

    def _execute(self, lesson_id, code):
        try:
            stage, index = self.lesson_location(lesson_id)
            lesson = COURSES[stage]["lessons"][index]
            missing = [name for name in COURSES[stage]["requirements"] if importlib.util.find_spec(name) is None]
            if missing:
                raise HTTPException(409, "本阶段需要安装依赖：python -m pip install " + " ".join(missing))
            self.prepare_kernel(stage, index)
            result = self.kernel.execute(code, timeout=lesson.get("timeout", 20))
            passed = False
            differences = []
            message = "运行出错了。读一读下面的错误，修改后可以重新运行。"
            if result["ok"]:
                if lesson["kind"] == "example":
                    comparison = compare_example(lesson["code"], code)
                    passed = comparison["matched"]
                    differences = comparison["differences"]
                    message = "运行成功，示例核对通过。可以进入下一段。" if passed else "运行成功，但与示例存在以下差异。核对已忽略空白和字符串中的中英文符号，请检查文字内容、变量、数值与代码结构。"
                else:
                    validation = self.kernel.execute("exec(" + repr(lesson["test"]) + ", dict(globals()))")
                    passed = validation["ok"]
                    result["outputs"].extend(validation["outputs"])
                    message = "运行与边界检查全部通过。你完成了这道练习。" if passed else "代码可以运行，但练习检查未通过。请核对题目要求与边界情况。"
            self.kernel.position = (stage, index + 1) if passed else None
            with self.connect() as database:
                database.execute(
                    "INSERT INTO lessons(id, draft, code, outputs) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET draft=excluded.draft, code=COALESCE(excluded.code, lessons.code), outputs=excluded.outputs",
                    (lesson_id, code, code if passed else None, json.dumps(result["outputs"], ensure_ascii=False)),
                )
            return {**result, "passed": passed, "message": message, "differences": differences, "state": self.state(stage)}
        except HTTPException:
            raise
        except Exception as error:
            self.kernel.close()
            raise HTTPException(500, f"内核连接失败：{error}。请检查是否安装 ipykernel 后重试。") from error

    def close(self):
        self.worker.submit(self.kernel.close).result()
        self.worker.shutdown(wait=True)


class UserLabs:
    """每个账号使用独立进度数据库、工作目录、执行锁和 Jupyter 内核。"""

    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.labs = {}

    def get(self, user_id):
        with self.lock:
            if user_id not in self.labs:
                self.labs[user_id] = Lab(self.root / "users" / user_id)
            return self.labs[user_id]

    def close_kernel(self, user_id):
        with self.lock:
            lab = self.labs.get(user_id)
        if lab:
            with lab.lock:
                lab.worker.submit(lab.kernel.close).result()

    def close(self):
        with self.lock:
            labs = list(self.labs.values())
        for lab in labs:
            lab.close()


def create_app(data_dir=None):
    root = Path(data_dir or ROOT / ".learning_lab")
    root.mkdir(parents=True, exist_ok=True)
    accounts = Accounts(root)
    labs = UserLabs(root)

    def current_lab(request):
        return labs.get(request.state.session["user"]["id"])

    def with_identity(state, request):
        return {**state, **request.state.session}

    @asynccontextmanager
    async def lifespan(app):
        yield
        labs.close()

    app = FastAPI(title="Transformer Lab", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.accounts = accounts
    app.state.labs = labs

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.url.hostname not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "服务仅供本机访问。"}, status_code=403)
        is_api = request.url.path.startswith("/api/")
        public_auth = request.url.path in ("/api/auth/session", "/api/auth/register", "/api/auth/login")
        session = accounts.session(request.cookies.get(COOKIE_NAME)) if is_api else None
        request.state.session = session
        if is_api and not public_auth:
            if not session:
                return JSONResponse({"detail": "请先登录，再继续学习。"}, status_code=401)
            requested_user = request.headers.get("x-lab-user")
            if requested_user and requested_user != session["user"]["id"]:
                return JSONResponse({"detail": "登录账号已变更，请刷新页面。", "code": "ACCOUNT_CHANGED"}, status_code=409)
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
                return JSONResponse({"detail": "不接受跨站请求。"}, status_code=403)
            expected_token = session["token"] if session else accounts.bootstrap_token
            if not secrets.compare_digest(request.headers.get("x-lab-token", "").encode("utf-8"), expected_token.encode("utf-8")):
                return JSONResponse({"detail": "请刷新学习页面后再试。"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/auth/session")
    def auth_session(request: Request):
        session = request.state.session
        if session:
            return {"authenticated": True, **session}
        return {"authenticated": False, "token": accounts.bootstrap_token, "first_account": accounts.first_account(), "legacy_available": accounts.legacy_available()}

    def authenticate_response(user, request, **extra):
        token, csrf = accounts.new_session(user)
        response = JSONResponse({"authenticated": True, "user": user, "token": csrf, **extra})
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict", secure=request.url.scheme == "https", path="/")
        return response

    @app.post("/api/auth/register")
    def register(body: Credentials, request: Request):
        if request.state.session:
            raise HTTPException(409, "请先退出当前账号，再注册其他账号。")
        user, claimed, imported = accounts.register(body.username, body.password, body.import_legacy)
        return authenticate_response(user, request, legacy_claimed=claimed, legacy_imported=imported)

    @app.post("/api/auth/login")
    def login(body: Credentials, request: Request):
        if request.state.session:
            raise HTTPException(409, "请先退出当前账号，再登录其他账号。")
        user = accounts.login(body.username, body.password, request.client.host if request.client else "local")
        return authenticate_response(user, request)

    @app.post("/api/auth/logout")
    def logout(request: Request):
        user_id = request.state.session["user"]["id"]
        accounts.logout(request.cookies[COOKIE_NAME])
        if not accounts.has_sessions(user_id):
            labs.close_kernel(user_id)
        response = JSONResponse({"message": "已退出登录，学习进度已保留。"})
        response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="strict")
        return response

    @app.get("/api/state")
    def state(request: Request, stage: int | None = None):
        return with_identity(current_lab(request).state(stage), request)

    @app.get("/api/lessons/{lesson_id}")
    def lesson(lesson_id: str, request: Request):
        return current_lab(request).lesson(lesson_id)

    @app.post("/api/draft")
    def draft(body: CodeRequest, request: Request):
        current_lab(request).save_draft(body.lesson_id, body.code)
        return {"saved": True}

    @app.post("/api/execute")
    def execute(body: CodeRequest, request: Request):
        result = current_lab(request).execute(body.lesson_id, body.code)
        result["state"] = with_identity(result["state"], request)
        return result

    @app.post("/api/kernel/restart")
    def restart(request: Request):
        lab = current_lab(request)
        if not lab.lock.acquire(blocking=False):
            raise HTTPException(409, "当前正在运行代码，请等待运行结束。")
        try:
            lab.worker.submit(lab.kernel.close).result()
            return {"message": "内核已重置。下一次运行会恢复当前阶段之前通过的代码；学习进度保留。"}
        finally:
            lab.lock.release()

    @app.get("/api/quiz")
    def quiz(request: Request, stage: int = 1):
        lab = current_lab(request)
        lab.require_stage(stage)
        if lab.frontier(stage=stage) != len(COURSES[stage]["lessons"]):
            raise HTTPException(403, "完成所有示例与练习后，才能进入阶段验收。")
        return {"questions": COURSES[stage]["quiz"], "stage": stage}

    @app.post("/api/quiz")
    def submit_quiz(body: QuizRequest, request: Request):
        lab = current_lab(request)
        lab.require_stage(body.stage)
        course = COURSES[body.stage]
        if lab.frontier(stage=body.stage) != len(course["lessons"]):
            raise HTTPException(403, "请先完成本阶段全部代码。")
        wrong = [index + 1 for index, (answer, expected) in enumerate(zip(body.answers, course["answers"])) if answer != expected]
        if wrong:
            return {"passed": False, "message": "请再想一想第 " + "、".join(map(str, wrong)) + " 题。可以返回示例复习后再答。"}
        with lab.connect() as database:
            database.execute("INSERT INTO metadata(key, value) VALUES (?, 'true') ON CONFLICT(key) DO UPDATE SET value='true'", (f"stage_{body.stage}_complete",))
        message = f"第 {body.stage} 阶段已完成！" + (f"第 {body.stage + 1} 阶段已解锁。" if body.stage < max(COURSES) else "你已完成全部十个阶段。")
        return {"passed": True, "message": message, "state": with_identity(lab.state(body.stage), request)}

    @app.get("/api/export")
    def export(request: Request, stage: int | None = None):
        lab = current_lab(request)
        stage = lab.resume_stage() if stage is None else stage
        lab.require_stage(stage)
        records = lab.records()
        cells = []
        lessons = COURSES[stage]["lessons"]
        for lesson in lessons[:min(lab.frontier(stage=stage) + 1, len(lessons))]:
            record = records.get(lesson["id"], {})
            cells.extend([
                {"cell_type": "markdown", "id": lesson["id"] + "-note", "metadata": {}, "source": f"## {lesson['title']}\n\n{lesson['description']}"},
                {"cell_type": "code", "id": lesson["id"] + "-code", "metadata": {}, "source": record.get("draft", ""), "execution_count": None, "outputs": []},
            ])
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}, "cells": cells}
        return Response(json.dumps(notebook, ensure_ascii=False), media_type="application/x-ipynb+json", headers={"Content-Disposition": f'attachment; filename="my-transformer-stage-{stage:02d}.ipynb"'})

    @app.get("/api/materials/pretrained")
    def pretrained_material(request: Request):
        current_lab(request).require_stage(9)
        return FileResponse(ROOT / "course_materials" / "stage_09" / "optional_pretrained.py", filename="optional_pretrained.py", media_type="text/x-python")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
