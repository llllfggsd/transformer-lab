"""在临时工作目录实际运行十个阶段的全部示例和参考解法。"""

from contextlib import redirect_stdout
import io
import os
from tempfile import TemporaryDirectory
import unittest

from webapp.curriculum import ALL_LESSONS, COURSES


class CurriculumContentTests(unittest.TestCase):
    def test_every_course_runs_offline_and_answers_pass(self):
        self.assertEqual(len(COURSES), 10)
        self.assertEqual(len({item["id"] for item in ALL_LESSONS}), len(ALL_LESSONS))
        original = os.getcwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for number, course in COURSES.items():
                    namespace = {"__name__": "__course_check__"}
                    with self.subTest(stage=number), redirect_stdout(io.StringIO()):
                        for lesson in course["lessons"]:
                            code = lesson["code"] if lesson["kind"] == "example" else lesson["solution"]
                            exec(compile(code, lesson["id"], "exec"), namespace)
                            if lesson["kind"] == "practice":
                                exec(compile(lesson["test"], lesson["id"] + "-checks", "exec"), dict(namespace))
                        self.assertEqual(len(course["quiz"]), len(course["answers"]))
                        for question, answer in zip(course["quiz"], course["answers"]):
                            self.assertIn(answer, range(len(question["options"])))
                        if number == 9:
                            self.assertLess(namespace["history"][-1], namespace["history"][0])
            finally:
                os.chdir(original)
