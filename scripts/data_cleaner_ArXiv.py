# data_cleaner.py

import os
import re
import pandas as pd
import pyarrow as pa
from datasets import load_dataset
from tqdm import tqdm

def clean_text(text):
    """去除 HTML 标签和乱码字符"""
    text = re.sub(r"<.*?>", "", text)  # 去除HTML标签
    text = re.sub(r"[^\w\s,.!?，。！？]", "", text)  # 去除非标准字符
    return text.replace("\n", " ").strip()  # 去掉换行符

def preprocess_data(data_dir):
    # 从 Hugging Face 加载数据集
    dataset = load_dataset("ccdv/arxiv-summarization")

    # 输出目录，用于保存预处理后的数据
    output_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        processed_data = []

        for entry in tqdm(dataset[split], desc=f"Processing {split} data"):
            # 检查摘要长度
            abstract = entry["abstract"]
            if len(abstract) > 9000:
                continue

            # 清理文本
            text = clean_text(entry["article"])
            abstract = clean_text(abstract)

            # 将清理后的数据添加到列表
            processed_data.append({"text": text, "abstract": abstract})

        # 转换为 Arrow 格式并保存
        table = pa.Table.from_pandas(pd.DataFrame(processed_data))
        table_path = os.path.join(output_dir, f"{split}_data.arrow")
        with pa.OSFile(table_path, 'wb') as sink:
            writer = pa.RecordBatchFileWriter(sink, table.schema)
            writer.write_table(table)
            writer.close()

        print(f"Saved {split} data to {table_path}")

if __name__ == "__main__":
    data_dir = "/usr1/home/s124mdg41_08/EE6405_D24/data/ArXiv"  # 使用绝对路径
    preprocess_data(data_dir)