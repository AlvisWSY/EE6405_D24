import subprocess
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(project_root, 'setup', script_name)
    try:
        subprocess.check_call([sys.executable, script_path])
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    print("Initializing the project...")

    # Step 1: 配置 conda 环境
    print("Step 1: Setting up conda environment...")
    run_script('setup_conda_env.py')

    # Step 2: 创建文件夹结构
    print("Step 2: Creating folder structure...")
    run_script('create_folders.py')

    # Step 3: 下载数据集
    print("Step 3: Downloading datasets...")
    run_script('download_datasets.py')

    print("Initialization complete!")