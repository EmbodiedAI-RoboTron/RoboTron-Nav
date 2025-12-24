import os
import glob
import zipfile
from concurrent.futures import ThreadPoolExecutor

if 0:
    abs_datasets_dir = '~/yanfeng/data/robotics/metaworld_uniview/'

    suffix='.zip'
    search_pattern =  os.path.join(abs_datasets_dir, '**', f'*{suffix}') # 构建搜索模式
    matching_files = glob.glob(search_pattern, recursive=True)
    zip_files = [file for file in matching_files] # 获取并返回这些文件的绝对路径

    def unzip_one(zip_path):
        print(zip_path)
        extract_to = os.path.splitext(zip_path.replace('/metaworld_uniview/', '/metaworld_uniview_unzip/'))[0]
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(unzip_one, zip_files)

if 1:
    abs_datasets_dir = '~/yanfeng/data/robotics/robocasa_uniview'

    suffix='.zip'
    search_pattern =  os.path.join(abs_datasets_dir, "**", f'*{suffix}') # 构建搜索模式
    matching_files = glob.glob(search_pattern, recursive=True)
    zip_files = [file for file in matching_files] # 获取并返回这些文件的绝对路径

    def unzip_one(zip_path):
        print(zip_path)
        extract_to = os.path.splitext(zip_path.replace('/robocasa_uniview/', '/robocasa_uniview_unzip/'))[0]
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(unzip_one, zip_files)