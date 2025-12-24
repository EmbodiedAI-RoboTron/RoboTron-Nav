import zipfile
import os

def unzip_files_in_folder(folder_path):
    # 列出文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 遍历所有文件
    for file in files:
        # 检查文件是否为 .zip 文件
        if file.endswith('.zip'):
            zip_path = os.path.join(folder_path, file)
            # 创建一个文件夹，用于存放解压后的文件
            extract_folder_name = os.path.splitext(file)[0]
            extract_folder_path = os.path.join(folder_path, extract_folder_name)
            os.makedirs(extract_folder_path, exist_ok=True)
            
            # 解压缩文件到指定文件夹
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder_path)
            
            print(f"{file} 解压缩到 {extract_folder_path}")

# 示例使用：
folder_path = '~/yanfeng/data/robotics/metaworld_uniview/'  # 替换为包含 .zip 文件的文件夹路径
files = os.listdir(folder_path)
for f in files:

    unzip_files_in_folder(os.path.join(folder_path, f))
