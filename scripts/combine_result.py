import os
from datasets import Dataset
import pandas as pd

# 定义输入和输出目录
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, '../output')
output_dir = os.path.join(base_dir, '../results')

os.makedirs(output_dir, exist_ok=True)

# 初始化数据集结构
datasets_combined = {}

# 获取方法目录列表
methods = [method for method in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, method))]

# 遍历每个方法
for method in methods:
    method_dir = os.path.join(input_dir, method)
    for dataset in os.listdir(method_dir):
        dataset_dir = os.path.join(method_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        # 找到数据集文件
        arrow_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith('.arrow')]
        if len(arrow_files) != 1:
            print(f"警告：{dataset_dir} 中未找到唯一的 Arrow 文件，跳过...")
            continue

        arrow_file = arrow_files[0]

        # 使用 Hugging Face 读取文件
        hf_dataset = Dataset.from_file(arrow_file)
        df = hf_dataset.to_pandas()

        # 如果是第一次处理该数据集
        if dataset not in datasets_combined:
            # 初始化数据集并保留 abstract 列
            datasets_combined[dataset] = pd.DataFrame({
                "abstract": df["abstract"]
            })

        # 添加当前方法的列
        datasets_combined[dataset][method] = df["output"]

# 保存合并后的数据集
for dataset_name, combined_df in datasets_combined.items():
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    output_file = os.path.join(dataset_output_dir, "combined.arrow")
    
    # 将 Pandas DataFrame 转换为 Hugging Face Dataset 并保存为 .arrow 文件
    hf_dataset = Dataset.from_pandas(combined_df)
    hf_dataset.save_to_disk(output_file)

print("文件合并完成！所有结果已保存在 result/ 目录下。")