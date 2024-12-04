import pandas as pd
from sklearn.model_selection import train_test_split


def convert_tsv_to_csv(input_file, output_file, rename_columns=None, drop_columns=None):
    """
    将 TSV 文件转换为 CSV 文件，并根据需要修改列名和删除指定列。

    Args:
        input_file (str): 输入的 TSV 文件路径。
        output_file (str): 输出的 CSV 文件路径。
        rename_columns (dict, optional): 列名修改字典，格式为 {'旧列名': '新列名'}。默认为 None。
        drop_columns (list, optional): 要删除的列名列表。默认为 None。
    """
    # 读取 TSV 文件
    df = pd.read_csv(input_file, sep='\t')

    # 修改列名
    if rename_columns:
        df.rename(columns=rename_columns, inplace=True)

    # 删除指定列
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)

    # 保存为 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"文件已成功保存为 {output_file}")


def split_csv_to_train_val_test(input_file, train_output, val_output, test_output, train_ratio=0.67, val_ratio=0.17,
                                random_state=None):
    """
    将 CSV 文件划分为训练集、验证集和测试集，并保存为三个新的 CSV 文件。

    Args:
        input_file (str): 输入的 CSV 文件路径。
        train_output (str): 训练集的输出文件路径。
        val_output (str): 验证集的输出文件路径。
        test_output (str): 测试集的输出文件路径。
        train_ratio (float): 训练集的比例，默认为 0.67（4:1:1 的划分）。
        val_ratio (float): 验证集的比例，默认为 0.17（4:1:1 的划分）。
        random_state (int, optional): 随机种子，便于复现结果。默认为 None。
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 首先划分出训练集和剩余数据集
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=random_state)

    # 然后在剩余数据集中划分验证集和测试集
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=random_state)

    # 保存训练集、验证集和测试集
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"训练集已保存为 {train_output}")
    print(f"验证集已保存为 {val_output}")
    print(f"测试集已保存为 {test_output}")


def extract_balanced_samples(csv_file_path, output_file_path, column_name="Y", sample_size=500):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 检查是否包含指定列
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the CSV file.")
        return

    # 从指定列中提取0和1的样本
    df_0 = df[df[column_name] == 0].sample(n=sample_size, random_state=42)
    df_1 = df[df[column_name] == 1].sample(n=sample_size, random_state=42)

    # 合并两个子集并打乱顺序
    balanced_df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

    # 保存到新文件
    balanced_df.to_csv(output_file_path, index=False)
    print(f"Saved balanced samples to {output_file_path}")


# 使用示例
# 将 "input.tsv" 文件转换为 "output.csv"，修改列名，并删除指定列
# convert_tsv_to_csv(
#     input_file='../datasets/bindingdb/drug_target_data_ec50.tsv',
#     output_file='../datasets/ec50.csv',
#     rename_columns={'Ligand SMILES': 'SMILES', 'BindingDB Target Chain Sequence': 'Protein','label':'Y'},
#     drop_columns=['smiles_sequence', 'amino_acid_sequence']
# )

split_csv_to_train_val_test(
    input_file='../datasets/ec50_1000.csv',
    train_output='../datasets/sample/random2/train.csv',
    val_output='../datasets/sample/random2/val.csv',
    test_output='../datasets/sample/random2/test.csv',
    train_ratio=0.67,  # 4:1:1 比例
    val_ratio=0.17,
    random_state=42
)
# extract_balanced_samples('../datasets/ec50.csv', '../datasets/ec50_1000.csv', sample_size=1000)
