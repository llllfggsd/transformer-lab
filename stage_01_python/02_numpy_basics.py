"""第一阶段 B：NumPy。重点观察每一步的 shape（形状）。"""

# %% 1. 数组和列表的区别
import numpy as np

python_list = [1, 2, 3]
vector = np.array([1, 2, 3], dtype=np.float32)
print("列表 * 2：", python_list * 2)  # 重复列表
print("数组 * 2：", vector * 2)  # 每个元素乘 2
print("向量形状：", vector.shape, "维数：", vector.ndim, "数据类型：", vector.dtype)

# %% 2. 二维数组：轴 0 是行，轴 1 是列
x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print("二维数组：\n", x)
print("shape：", x.shape, "ndim：", x.ndim, "size：", x.size)
print("第 0 行：", x[0], "shape：", x[0].shape)
print("保留二维结构的第 0 行：", x[0:1], "shape：", x[0:1].shape)
print("第 1 列：", x[:, 1])
print("第 1 行第 2 列：", x[1, 2])

# %% 3. 创建数组、改变形状、转置
print("全零数组：\n", np.zeros((2, 3), dtype=np.float32))
print("全一数组：\n", np.ones((2, 3), dtype=np.float32))
flat = np.arange(6)
reshaped = flat.reshape(2, 3)  # 元素总数必须相同：6 = 2 * 3
print("原数组：", flat, "reshape 后：\n", reshaped)
print("x 转置后：\n", x.T, "shape：", x.T.shape)

# %% 4. 基本运算、按轴求平均、广播
print("每个元素加 10：\n", x + 10)
print("每个元素平方：\n", x ** 2)
print("所有元素的平均值：", x.mean())
print("每列平均值（沿轴 0 汇总）：", x.mean(axis=0))
print("每行平均值（沿轴 1 汇总）：", x.mean(axis=1))

bias = np.array([10, 20, 30], dtype=np.float32)
print("x + bias：\n", x + bias)
# (2, 3) + (3,)：从最后一个维度对齐，bias 自动应用于每一行。
# 广播要求对齐的每个维度大小相等，或者其中一个为 1。

# %% 5. 区分逐元素乘法 * 和矩阵乘法 @
print("逐元素乘法 x * x：\n", x * x, "shape：", (x * x).shape)
weights = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
output = x @ weights
print("矩阵乘法 x @ weights：\n", output)
print("形状变化：", x.shape, "@", weights.shape, "->", output.shape)
# (2, 3) @ (3, 2) -> (2, 2)，中间的 3 必须相同。
# 左上角 = 1*1 + 2*0 + 3*1 = 4；请自己手算其余三个位置。

# %% 6. 批处理：把多个样本放到一个数组中
# 假设每行是一个样本的 3 个特征，一次处理两个样本。
batch = x
output_bias = np.array([0.5, -0.5], dtype=np.float32)
batch_output = batch @ weights + output_bias
print("batch shape：", batch.shape)
print("批处理结果：\n", batch_output, "shape：", batch_output.shape)

# 显式固定随机种子，让本示例的随机数据可重复。
rng = np.random.default_rng(42)
random_features = rng.normal(size=(2, 3)).astype(np.float32)
print("随机特征：\n", random_features)

# 下一阶段会解释矩阵乘法的数学意义；这里先掌握运算符与形状。
