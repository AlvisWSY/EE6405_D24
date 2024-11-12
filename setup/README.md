# Setup Directory

This directory contains scripts for creating formatted directories, downloading datasets, and setting up the environment.

## Scripts

### 1. `create_folders.py`

- **Description**: Creates the necessary directory structure for the project.
- **Functionality**:
  - Creates directories such as `data`, `src`, `models`, `scripts`, `results`, `gui`, `docs`.
  - Skips creation if the directory already exists.

### 2. `download_datasets.py`

- **Description**: Downloads the datasets required for the project.
- **Functionality**:
  - Downloads `cnn_dailymail` and `ArXiv` datasets from Hugging Face.
  - Downloads `article_data` dataset from a specified URL.
  - Saves the datasets to the `data/origin` directory.

### 3. `setup_conda_env.py`

- **Description**: Sets up the Conda environment.
- **Functionality**:
  - Creates a Conda environment named `NLP` based on the `environment.yml` file.
  - Outputs an error message if the environment setup fails.

## Usage

1. **Create Directories**:
   - Run `create_folders.py` to create the necessary directory structure.

2. **Download Datasets**:
   - Run `download_datasets.py` to download the required datasets.

3. **Set Up Conda Environment**:
   - Run `setup_conda_env.py` to create the Conda environment based on the `environment.yml` file.

By running these scripts, you can quickly set up and configure the project environment for further development and testing.

# Setup 目录

该目录包含用于创建格式化目录、下载数据集以及创建环境的脚本。

## 脚本

### 1. `create_folders.py`

- **描述**：创建项目所需的目录结构。
- **功能**：
  - 创建 `data`, `src`, `models`, `scripts`, `results`, `gui`, `docs` 等目录。
  - 如果目录已存在，则跳过创建。

### 2. `download_datasets.py`

- **描述**：下载项目所需的数据集。
- **功能**：
  - 从 Hugging Face 下载 `cnn_dailymail` 和 `ArXiv` 数据集。
  - 从指定 URL 下载 `article_data` 数据集。
  - 将数据集保存到 `data/origin` 目录。

### 3. `setup_conda_env.py`

- **描述**：创建 Conda 环境。
- **功能**：
  - 根据 `environment.yml` 文件创建名为 `NLP` 的 Conda 环境。
  - 如果环境创建失败，输出错误信息。

## 使用说明

1. **创建目录**：
   - 运行 `create_folders.py` 创建项目所需的目录结构。

2. **下载数据集**：
   - 运行 `download_datasets.py` 下载项目所需的数据集。

3. **创建 Conda 环境**：
   - 运行 `setup_conda_env.py` 根据 `environment.yml` 文件创建 Conda 环境。

通过运行这些脚本，可以快速设置和配置项目环境，方便后续开发和测试。
