from prettytable import PrettyTable
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# 从文本文件读取数据
with open("../output/result/sample/random1/test_markdowntable.txt", "r") as f:
    pretty_table_string = f.read()

# print(pretty_table_string)
# 解析 PrettyTable 字符串到 pandas DataFrame
data = StringIO(pretty_table_string)
df = pd.read_csv(data, sep='|', skipinitialspace=True, skiprows=1)

# 打印 DataFrame 的列名，检查列
print("Columns before dropping:", df.columns)

# 去掉第一列和最后一列
df.columns = df.columns.str.strip()  # 去除列名的空格
df = df.iloc[:, 1:-1]  # 选择从第二列到倒数第二列的数据
df = df.dropna()# 去掉包含 NaN 的每一行

# 确保数据为数值类型
df = df.apply(pd.to_numeric, errors='ignore')

# 打印 DataFrame
print(df)

# 绘制每一列的图表
for column in df.columns:  # 直接遍历 DataFrame 的每一列
    plt.figure(figsize=(10, 6))
    plt.plot(df.index + 1, df[column], marker='', label=column)  # 使用索引作为 X 轴数据
    plt.title(f'{column}')
    plt.xlabel('Epoch')
    plt.ylabel(column)
    # 设置 Y 轴范围为 0 到 1
    # plt.xticks(ticks=df.index + 1, labels=df["# Best Epoch"], rotation=45)  # 使用实际的 epoch 标签
    plt.legend()
    plt.grid()
    plt.savefig(f'../../image/output/random1/{column}.png', bbox_inches='tight')  # 保存为 PNG 文件
    plt.show()
    # 保存图像
    plt.close()  # 关闭图像以释放内存
