# Results Directory

This directory stores the summaries generated using different methods on two datasets. It contains two subdirectories, each representing one dataset. The data is stored in the `.arrow` format.

## Subdirectories

- `ArXiv/`
  - Contains the results for the ArXiv dataset.
  - Example file: [data-00000-of-00001.arrow](https://github.com/AlvisWSY/EE6405_D24/blob/4dd42c4c5fe67b1db838493b7903c9e92da12c03/results/ArXiv/data-00000-of-00001.arrow)
  
- `cnn_dailymail/`
  - Contains the results for the CNN/DailyMail dataset.
  - Example file: [data-00000-of-00001.arrow](https://github.com/AlvisWSY/EE6405_D24/blob/4dd42c4c5fe67b1db838493b7903c9e92da12c03/results/cnn_dailymail/data-00000-of-00001.arrow)

## Data Format

- The data is stored in the `.arrow` format using the Hugging Face library.
- The `abstract` attribute represents the ground truth summaries.
- Other attributes represent different generation methods.

# 结果目录

该目录存储了使用不同方法在两个数据集上生成的摘要。它包含两个子目录，每个子目录代表一个数据集。数据以 `.arrow` 格式存储。

## 子目录

- `ArXiv/`
  - 包含 ArXiv 数据集的结果。
  - 示例文件: [data-00000-of-00001.arrow](https://github.com/AlvisWSY/EE6405_D24/blob/4dd42c4c5fe67b1db838493b7903c9e92da12c03/results/ArXiv/data-00000-of-00001.arrow)
  
- `cnn_dailymail/`
  - 包含 CNN/DailyMail 数据集的结果。
  - 示例文件: [data-00000-of-00001.arrow](https://github.com/AlvisWSY/EE6405_D24/blob/4dd42c4c5fe67b1db838493b7903c9e92da12c03/results/cnn_dailymail/data-00000-of-00001.arrow)

## 数据格式

- 数据使用 Hugging Face 库以 `.arrow` 格式存储。
- `abstract` 属性表示摘要真值（即 ground truth）。
- 其他属性表示不同的生成方法。
