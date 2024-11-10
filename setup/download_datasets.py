import os
import requests
from datasets import load_dataset

def download_from_url(url, destination_folder, filename=None):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    if filename is None:
        filename = url.split("/")[-1]

    filepath = os.path.join(destination_folder, filename)
    
    # Stream the content and save it to the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"Downloaded {filename} to {destination_folder}")

def download_from_hf(dataset_name, subset, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dataset = load_dataset(dataset_name, subset)
    dataset.save_to_disk(destination_folder)
    print(f"Hugging Face dataset {dataset_name} (subset: {subset}) downloaded to {destination_folder}")

if __name__ == "__main__":
    # 获取当前脚本的绝对路径，确保文件下载到项目根目录的 data 文件夹中
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    destination_dir = os.path.join(project_root, 'data\origin')

    # 数据集信息字典
    datasets = {
        "cnn_dailymail": {
            "type": "hf",
            "name": "abisee/cnn_dailymail",
            "subset": "3.0.0"
        },
        "ArXiv": {
            "type": "hf",
            "name": "ccdv/arxiv-summarization",
            "subset": None  # ArXiv 没有特定子集，这里设为 None
        },
        "article_data": {
            "type": "url",
            "url": "https://example.com/article_data.zip",
            "filename": "article_data.zip"
        }
    }

    # 下载数据集
    for key, info in datasets.items():
        folder_path = os.path.join(destination_dir, key)
        if info["type"] == "hf":
            print(f"Downloading {key} dataset from Hugging Face...")
            download_from_hf(info["name"], info["subset"], folder_path)
        elif info["type"] == "url":
            print(f"Downloading {key} dataset from URL...")
            download_from_url(info["url"], folder_path, filename=info.get("filename"))

    print("All datasets downloaded successfully.")
