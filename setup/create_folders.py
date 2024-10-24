import os

def create_folders():
    # 获取当前目录，确保文件夹都创建在项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    directories = [
        "data",               # Directory for storing datasets
        "src",                # Source code directory
        "models",             # Directory for trained models
        "scripts",            # Directory for automation scripts
        "results",            # Directory for output results
        "gui",                # GUI related code and assets
        "docs",               # Documentation and project notes
    ]
    
    for directory in directories:
        path = os.path.join(project_root, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {directory} created at {path}.")
        else:
            print(f"Directory {directory} already exists at {path}.")

if __name__ == "__main__":
    create_folders()
