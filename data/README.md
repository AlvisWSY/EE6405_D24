# Data Directory Description

## Project Overview
This is the data directory for a Natural Language Processing (NLP) project. The dataset will be used for training and evaluating models.

## Dataset Description
The dataset is sourced from Hugging Face and includes the original ArXiv and CNN/DailyMail datasets. Detailed information about the datasets can be found at the following links:
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [ArXiv Dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization)

## Data Download and Preparation
Due to the large size of the dataset, it is not directly included in the repository. You can use the following steps to automatically download the data using a script:

1. Ensure the necessary dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script to download the data:
   ```bash
   python script/create_data.py
   ```

## Directory Structure
After downloading and preprocessing the data, the `data` directory structure is as follows:

```
data/
├── origin/
│   ├── ArXiv/
│   ├── cnn_dailymail/
├── processed/
│   ├── ArXiv/
│   ├── cnn_dailymail/
```

Both the original and processed datasets contain `test`, `train`, and `validation` directories, along with corresponding `.arrow` files and `dataset.json` files.

## Data Preprocessing
The data preprocessing script (`data_cleaner`) is located in the project's `script` directory. It performs various data cleaning operations such as removing special characters, placeholders, and HTML tags. The preprocessed data is saved in the `processed` directory with the same structure and format as the original datasets (in `.arrow` format).

## Usage Instructions
Example code for using the dataset in the project:
```python
from datasets import load_from_disk

def load_and_print_dataset(path, split_name):
    dataset = load_from_disk(path)
    print(f"Loaded {split_name}: {len(dataset)} samples")
    return dataset
```

## Contact Information
If you have any questions, please contact the project maintainer: AlvisWSY (GitHub username).



# 数据目录说明

## 项目简介
这是一个用于自然语言处理（NLP）项目的数据目录。数据集将用于训练和评估模型。

## 数据集描述
数据集来源于 Hugging Face 网站，包含 ArXiv 和 CNN/DailyMail 的原始数据集。数据集的详细信息可以在以下链接中找到：
- [CNN/DailyMail 数据集](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [ArXiv 数据集](https://huggingface.co/datasets/ccdv/arxiv-summarization)

## 数据下载和准备
数据集由于较大，没有直接包含在仓库中。可以使用以下步骤通过脚本自动下载数据：

1. 确保已安装必要的依赖项：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行脚本下载数据：
   ```bash
   python script/create_data.py
   ```

## 目录结构
下载和预处理数据后，`data` 目录的结构如下所示：

```
data/
├── origin/
│   ├── ArXiv/
│   ├── cnn_dailymail/
├── processed/
│   ├── ArXiv/
│   ├── cnn_dailymail/
```

原始数据集和预处理数据集均包含 `test`、`train` 和 `validation` 目录及相应的 `.arrow` 文件和 `dataset.json` 文件。

## 数据预处理
数据预处理脚本（data_cleaner）在项目目录的 `script` 目录下。分别对数据集进行了处理，例如，去特殊符号，占位符，html标签等操作。然后以原始数据集的相同结构、相同格式（.arrow）保存在 `processed` 目录下。

## 使用说明
在项目中使用数据集的示例代码：
```python
from datasets import load_from_disk

def load_and_print_dataset(path, split_name):
    dataset = load_from_disk(path)
    print(f"Loaded {split_name}: {len(dataset)} samples")
    return dataset
```

## 联系信息
如有任何问题，请联系项目维护者：AlvisWSY (GitHub用户名).
