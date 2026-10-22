# 从 Python 开始学习 Transformer

十个阶段的课程已补齐，共 **81 段示例、35 道独立练习、30 道阶段验收题**。每阶段包含中文讲解与可运行代码；网页按账号保存进度，完成当前阶段的代码和验收后解锁下一阶段。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 01 | [Python、NumPy 与 Jupyter](stage_01_python/README.md) | 已补齐 |
| 02 | [必要的数学基础](course_materials/stage_02/README.md) | 已补齐，顺序解锁 |
| 03 | [机器学习基础](course_materials/stage_03/README.md) | 已补齐，顺序解锁 |
| 04 | [深度学习与 PyTorch](course_materials/stage_04/README.md) | 已补齐，顺序解锁 |
| 05 | [文本如何变成模型输入](course_materials/stage_05/README.md) | 已补齐，顺序解锁 |
| 06 | [注意力机制](course_materials/stage_06/README.md) | 已补齐，顺序解锁 |
| 07 | [Transformer 完整结构](course_materials/stage_07/README.md) | 已补齐，顺序解锁 |
| 08 | [常见模型与训练方式](course_materials/stage_08/README.md) | 已补齐，顺序解锁 |
| 09 | [动手实践](course_materials/stage_09/README.md) | 已补齐，顺序解锁 |
| 10 | [进阶内容](course_materials/stage_10/README.md) | 已补齐，顺序解锁 |

## 网页学习工作台

现在可以在本机网页中学习：左侧是示例与讲解，右侧连接真实 Jupyter 内核，手敲后逐段运行。禁止粘贴、拖入和跳段；完成示例、练习及阶段验收后，才满足下一阶段的开启条件。

在 IDE 中打开项目根目录的 [main.py](main.py)，点击“运行”即可启动服务，浏览器会自动打开学习页面。端口等启动配置也直接在 `main.py` 中修改。

首次使用如果缺少依赖，在所选 Python 环境中安装：

```powershell
python -m pip install -r webapp\requirements.txt
```

默认地址是 **http://127.0.0.1:8765**。保持程序运行即可学习；停止时使用 IDE 的停止按钮或在终端按 `Ctrl+C`。

首次打开先注册账号，之后通过用户名和密码登录。每个账号分别保存进度、草稿、运行结果和阶段验收记录；右上角显示当前用户名并提供退出入口。已有的本机记录可以在首次注册时选择导入，后续账号独立开始。

网页已接入全部十个阶段。第一阶段仍保留原有 13 段示例和 8 道练习，已有账号数据无需清空；完成验收后可以点击“进入下一阶段”，也可回看已完成章节。详细规则见 [网页使用说明](webapp/README.md)。

完整的 [文件版课程目录](course_materials/README.md) 提供各章节 Notebook、示例脚本、练习和参考答案。主要课程使用小数据和 CPU，不需要下载大模型；第 9 阶段另外提供可选的真实预训练模型下载实践。

也可以从 [第一阶段文件版学习说明](stage_01_python/README.md) 开始。不要求提前掌握数学或机器学习。
