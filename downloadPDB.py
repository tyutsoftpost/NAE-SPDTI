import pandas as pd
import requests
import os


def download_pdb_files_from_csv(csv_file_path, column_name="PDB ID(s) of Target Chain", download_folder="../datasets/pdb_files"):
    # 创建下载文件夹
    os.makedirs(download_folder, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 获取指定列并去重
    unique_ids = set()
    for pdb_id_list in df[column_name].dropna():
        ids = pdb_id_list.split(',')
        unique_ids.update(ids)

    # 下载PDB文件
    for pdb_id in unique_ids:
        pdb_id = pdb_id.strip()  # 去除空白字符
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)

        if response.status_code == 200:
            with open(os.path.join(download_folder, f"{pdb_id}.pdb"), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdb_id}.pdb")
        else:
            print(f"Failed to download: {pdb_id}.pdb")


# 示例调用
csv_file_path = "path/to/your/file.csv"  # 请替换为你的CSV文件路径
download_pdb_files_from_csv(csv_file_path)
