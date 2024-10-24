import subprocess
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_conda_environment():
    # Conda 环境的名称
    conda_env_name = "NLP"

    # 获取根目录下的 environment.yml 文件路径
    environment_file = os.path.join(project_root, 'environment.yml')

    # 创建 conda 环境的命令，使用 environment.yml 文件
    create_env_command = [
        'conda', 'env', 'create', '-f', environment_file, '--name', conda_env_name
    ]
    
    try:
        # 运行创建 conda 环境的命令
        subprocess.check_call(create_env_command)
        print(f"Conda environment '{conda_env_name}' created successfully from environment.yml.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error setting up conda environment: {e}")

if __name__ == "__main__":
    setup_conda_environment()
