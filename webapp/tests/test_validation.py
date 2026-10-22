import unittest

from webapp.curriculum import LESSONS
from webapp.validation import compare_example


class ExampleComparisonTests(unittest.TestCase):
    def test_layout_comments_quotes_and_blank_lines_are_ignored(self):
        self.assertTrue(compare_example('name = "小林"\nprint(name)', "# 注释\n\nname='小林'\n\nprint( name )\n")["matched"])

    def test_screenshot_spacing_and_punctuation_are_accepted(self):
        expected = LESSONS[0]["code"]
        actual = expected.replace(" = ", "=").replace("，今天计划学习 ", ",今天计划学习").replace("} 分钟", "}分钟")
        result = compare_example(expected, actual)
        self.assertTrue(result["matched"])
        self.assertEqual(result["differences"], [])

    def test_string_whitespace_and_symbols_are_ignored(self):
        for expected, actual in [
            ('text="你好， 世界！"', 'text="你好,世界!"'),
            ('text="学习 30 分钟。"', 'text="学习30分钟"'),
            ('text="a\\tb\\nc"', 'text="abc"'),
            ('text="【hello】（world）+="', 'text="helloworld"'),
            ('text="， 。 !"', 'text=""'),
            ('print(f" {name}! ")', 'print(f"{name}")'),
            ('print(f"，{name}。{minutes}！")', 'print(f"{name}{minutes}")'),
        ]:
            with self.subTest(actual=actual):
                self.assertTrue(compare_example(expected, actual)["matched"])

    def test_string_words_and_fstring_expressions_are_still_checked(self):
        for expected, actual in [
            ('text="学习分钟数："', 'text="学习秒数:"'),
            ('text="hello world"', 'text="hello word"'),
            ('text="学习30分钟"', 'text="学习3分钟"'),
            ('print(f"你好，{learner}!")', 'print(f"你好，{learners}!")'),
        ]:
            with self.subTest(actual=actual):
                result = compare_example(expected, actual)
                self.assertFalse(result["matched"])
                self.assertTrue(result["differences"])

    def test_blank_lines_do_not_shift_reported_reference_location(self):
        result = compare_example('text = "hello world"', '\n\ntext="hello word"')
        difference = result["differences"][0]
        self.assertEqual(difference["line"], 3)
        self.assertEqual(difference["reference_line"], 1)

    def test_constants_names_operators_and_missing_statements(self):
        for expected, actual in [('x = True', 'x = 1'), ('x = 30', 'x = 3'), ('name = 1', 'names = 1'), ('x=1+2', 'x=1-2'), ('x=1\nprint(x)', 'x=1')]:
            with self.subTest(actual=actual):
                result = compare_example(expected, actual)
                self.assertFalse(result["matched"])
                self.assertTrue(result["differences"])

    def test_unicode_columns_use_characters_instead_of_utf8_bytes(self):
        result = compare_example('print("你好", "x y")', 'print("你好", "xyy")')
        self.assertEqual(result["differences"][0]["column"], 13)
        self.assertEqual(result["differences"][0]["expected"], "x y")


if __name__ == "__main__":
    unittest.main()
