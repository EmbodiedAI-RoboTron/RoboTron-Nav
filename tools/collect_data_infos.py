import os, json, pickle, h5py
import numpy as np
from pathlib import Path
from collections import Counter

languages = {}
bins = {"pick": 0,
        "place": 0,
        "put": 0,
        "push": 0,
        "turn": 0,
        "grasp": 0,
        "stack": 0,
        "close": 0,
        "rotate": 0,
        "move": 0,
        "open": 0,
        "use": 0,}
bins = Counter(bins)

# calvin
abs_datasets_dir = Path('~/yanfeng/data/robotics/task_D_D/training')
data_path = '~/liufanfan-mv-from/data/new_calvin_D_1/training'
lang_data = np.load(abs_datasets_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True,).item()

ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64, me
lang_ann = lang_data["language"]["ann"]  # length total number of annotations
for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
    bins.update([k for k in bins if k in lang_ann[i]])

lang_ann = set(lang_ann)
languages['calvin'] = lang_ann

# metaworld
abs_datasets_dir = '~/yanfeng/data/robotics/metaworld_uniview/'
json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
with open(json_path, 'r') as f:
    data = json.load(f)
lang_ann = []
for data_slice in data:
    lang_ann.append(data_slice[3])
    bins.update([k for k in bins if k in data_slice[3]])
lang_ann = set(lang_ann)
languages['metaworld'] = lang_ann

# libero
lang_ann = []
abs_datasets_dir = Path('~/yanfeng/data/robotics/libero_uniview_final')
task_emb_list = np.load(abs_datasets_dir/'emb.npz', allow_pickle=True)
_task_list = os.listdir(abs_datasets_dir)
for task in _task_list:
    data_dir = abs_datasets_dir / task
    if not data_dir.is_dir(): continue # 样本均在文件夹内，假如非文件夹则无效
    lang_ann.append(task_emb_list['arr_0'][int(task.split('_')[-1])])
    bins.update([k for k in bins if k in task_emb_list['arr_0'][int(task.split('_')[-1])]])
lang_ann = set(lang_ann)
languages['libero'] = lang_ann

# robomimic
lang_ann = []
abs_datasets_dir = '~/yanfeng/data/robotics/robomimic_uniview'
_task_list = os.listdir(abs_datasets_dir)
_task_list = ["lift", "can", "square"]# , "tool_hang"
lang_ann_ = {
            "lift" : "The robot arm must lift a small cube.",
            "can" : "The robot must place a coke can from a large bin into a smaller target bin.",
            "square" : "The robot must pick a square nut and place it on a rod.",
            "transport" : "Two robot arms must transfer a hammer from a closed container on a shelf to a target bin on another shelf.",
            "tool_hang" : "Insert the hook into the base, assemble the frame, and hang the wrench on the hook.",
        }
for task in _task_list:
    lang_ann.append(lang_ann_[task])
    dataset_file = os.path.join(abs_datasets_dir, task, "ph", "depth.hdf5")
    data = h5py.File(dataset_file, "r")
    episodes = data["data"]             # 记录每一条轨迹的信息
    for i in range(len(episodes)):
        bins.update([k for k in bins if k in lang_ann_[task]])

lang_ann = set(lang_ann)
languages['robomimic'] = lang_ann

# robocasa
lang_ann = []
abs_datasets_dir = '~/yanfeng/data/robotics/robocasa_uniview'
json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
with open(json_path, 'r') as f:
    data = json.load(f)
for data_slice in data:
    lang_ann.append(data_slice[3])
    bins.update([k for k in bins if k in data_slice[3]])
lang_ann = set(lang_ann)
languages['robocasa'] = lang_ann

# maniskill2
lang_ann = []
abs_datasets_dir = '~/yanfeng/data/robotics/maniskill2_uniview'
task_emb_list = np.load(os.path.join(abs_datasets_dir, 'maniskill2_lang.py.npy'), allow_pickle=True).item()
abs_datasets_dir = os.path.join(abs_datasets_dir, "v0")
def find_files_with_suffix(root_path, suffix):
    import glob
    search_pattern = os.path.join(root_path, '**', f'*{suffix}') # 构建搜索模式
    matching_files = glob.glob(search_pattern, recursive=True) # 在指定路径下递归搜索所有符合后缀的文件
    absolute_paths = [os.path.abspath(file) for file in matching_files] # 获取并返回这些文件的绝对路径
    return absolute_paths
h5_files = find_files_with_suffix(abs_datasets_dir, 'rgbd.pd_ee_delta_pose.h5')
for h5_f in h5_files:
    task = [elem for elem in list(task_emb_list.keys()) if elem in h5_f]
    lang_ann.append(task_emb_list[task[0]])
    json_path = h5_f.replace(".h5", ".json")
    json_data = json.load(open(json_path, 'rb'))
    episodes = json_data["episodes"]
    for i in range(len(episodes)):
        bins.update([k for k in bins if k in task_emb_list[task[0]]])
lang_ann = set(lang_ann)
languages['maniskill2'] = lang_ann

# rlbench
lang_ann = []
abs_datasets_dir = Path('~/robotics/RLbench/RT/train')
tasks=[x for x in "place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap".split(" ")]
for task_str in tasks:
    task_path = abs_datasets_dir / task_str / "all_variations" / "episodes"
    for num_demo in os.listdir(task_path):
        task_name = f"{task_str}/all_variations/episodes/{num_demo}"
        episodes_dir = abs_datasets_dir / task_name
        with open(episodes_dir/"variation_descriptions.pkl", 'rb') as f:
            lang_str = pickle.load(f)
        # if self.only_single_task and "3" not in lang_str[0]: continue # 测试使用
        lang_ann.append(lang_str[0])
        bins.update([k for k in bins if k in lang_str[0]])
lang_ann = set(lang_ann)
languages['rlbench'] = lang_ann

# colosseum
lang_ann = []
abs_datasets_dir = Path('~/huangyiyang02/data/robot-colsseum')
tasks = os.listdir(abs_datasets_dir)
for task_str in tasks:
    task_path = abs_datasets_dir / task_str / "variation0" / "episodes"
    for num_demo in os.listdir(task_path):
        task_name = f"{task_str}/variation0/episodes/{num_demo}"
        episodes_dir = abs_datasets_dir / task_name
        with open(abs_datasets_dir/ task_str / "variation0" / "variation_descriptions.pkl", 'rb') as f:
            lang_str = pickle.load(f)
        # if self.only_single_task and "3" not in lang_str[0]: continue # 测试使用
        lang_ann.append(lang_str[0])
        bins.update([k for k in bins if k in lang_str[0]])
lang_ann = set(lang_ann)
languages['colosseum'] = lang_ann

import re
from collections import Counter

def preprocess_text(text):
    # 将文本转换为小写
    text = text.lower()
    # 使用正则表达式去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去除多余的空格
    text = text.strip()
    return text

def get_word_frequencies(text):
    # 预处理文本
    processed_text = preprocess_text(text)
    # 分词
    words = processed_text.split()
    # 统计词频
    word_counts = Counter(words)
    # 按频率高低排序并返回所有单词及其频率
    sorted_word_counts = word_counts.most_common()
    return sorted_word_counts

# 获取按频率排序的单词及其频率
sorted_word_counts = get_word_frequencies(' '.join([' '.join(list(languages[k])) for k in languages]))
print("Words sorted by frequency:")
for word, freq in sorted_word_counts:
    print(f"'{word}': {freq}")

print(bins)
# Counter({'place': 4335, 'pick': 3494, 'turn': 3452, 'stack': 1654, 'push': 1503, 'grasp': 1196, 'close': 1170, 'put': 1091, 'open': 688, 'rotate': 583, 'move': 485, 'use': 200})