"""忽略字符串空白和符号后核对语法结构，并返回可定位的差异。"""

import ast
import unicodedata


class ExampleNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node._original_string = node.value
            node.value = "".join(character for character in node.value if not character.isspace() and unicodedata.category(character)[0] not in ("P", "S"))
        return node

    def visit_JoinedStr(self, node):
        self.generic_visit(node)
        # f" {name}!" 和 f"{name}" 规范化后应一致，去除仅含空白/符号的片段。
        node.values = [value for value in node.values if not (isinstance(value, ast.Constant) and value.value == "")]
        return node


def compare_example(expected_source, actual_source):
    expected_tree = ast.parse(expected_source)
    try:
        actual_tree = ast.parse(actual_source)
    except SyntaxError as error:
        return {"matched": False, "differences": [{"line": error.lineno or 1, "reference_line": 1, "column": error.offset or 1, "message": "请使用与示例对应的 Python 代码结构。", "expected": "合法的 Python 示例", "actual": error.msg}]}
    expected_tree = ExampleNormalizer().visit(expected_tree)
    actual_tree = ExampleNormalizer().visit(actual_tree)
    if ast.dump(expected_tree) == ast.dump(actual_tree):
        return {"matched": True, "differences": []}

    differences = []

    def describe(value):
        if isinstance(value, ast.AST):
            try:
                return ast.unparse(value)
            except (ValueError, TypeError):
                return type(value).__name__
        return value if isinstance(value, str) else repr(value)

    def add(expected, actual, expected_node, actual_node, message):
        # 核对采用规范化内容，差异提示仍展示用户输入的原文。
        if isinstance(expected, str) and isinstance(actual, str) and isinstance(expected_node, ast.Constant) and isinstance(actual_node, ast.Constant):
            expected = getattr(expected_node, "_original_string", expected)
            actual = getattr(actual_node, "_original_string", actual)
        line = getattr(actual_node, "lineno", 1)
        byte_column = getattr(actual_node, "col_offset", 0)
        text = actual_source.splitlines()[line - 1] if line <= len(actual_source.splitlines()) else ""
        column = len(text.encode("utf-8")[:byte_column].decode("utf-8", errors="ignore")) + 1
        differences.append({"line": line, "reference_line": getattr(expected_node, "lineno", 1), "column": column, "message": message, "expected": describe(expected)[:300], "actual": describe(actual)[:300]})

    def visit(expected, actual, expected_parent, actual_parent, field=""):
        if len(differences) >= 5:
            return
        if isinstance(expected, ast.AST) and isinstance(actual, ast.AST):
            expected_parent = expected if hasattr(expected, "lineno") else expected_parent
            actual_parent = actual if hasattr(actual, "lineno") else actual_parent
            if type(expected) is not type(actual):
                add(expected, actual, expected_parent, actual_parent, "代码结构不同。")
                return
            for name, value in ast.iter_fields(expected):
                visit(value, getattr(actual, name), expected_parent, actual_parent, name)
        elif isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                # 缺少语句时优先定位到最后一条已写语句，便于继续补齐。
                anchor = actual[-1] if actual and isinstance(actual[-1], ast.AST) else actual_parent
                reference = expected[-1] if expected and isinstance(expected[-1], ast.AST) else expected_parent
                add(f"{len(expected)} 项", f"{len(actual)} 项", reference, anchor, "语句、参数或字符串片段的数量不同，请检查是否漏写或多写。")
            for left, right in zip(expected, actual):
                visit(left, right, expected_parent, actual_parent, field)
        elif type(expected) is not type(actual) or expected != actual:
            message = "代码内容不同。"
            if isinstance(expected_parent, ast.Constant) and isinstance(expected, str):
                message = "字符串文字内容不同（已忽略空白和中英文符号）。"
            elif field in ("id", "arg", "name", "attr"):
                message = "名称不同，请核对变量、函数、参数或属性的拼写。"
            elif isinstance(expected_parent, ast.Constant):
                message = "数值或常量不同。"
            add(expected, actual, expected_parent, actual_parent, message)

    visit(expected_tree, actual_tree, expected_tree, actual_tree)
    return {"matched": False, "differences": differences[:5]}
