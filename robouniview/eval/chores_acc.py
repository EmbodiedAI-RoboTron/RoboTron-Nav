import os
import glob
import argparse

def calculate_succ_rate_and_average_score_and_percentage_room(dir_path):
    # 查找所有gif文件
    gif_files = glob.glob(os.path.join(dir_path, "*.gif"))
    total_files = len(gif_files)  # 总文件数量
    num_succ = 0  # 统计含有'succ'的文件数量
    total_score = 0.0  # 所有成功文件的分数之和
    total_room = 0.0

    for gif_file in gif_files:
        file_name = os.path.basename(gif_file)
        is_succ = file_name.split('.gif')[0].split('_')[-3]
        if is_succ == 'succ':
            num_succ += 1
            # 提取并累加分数
            try:
                score_str = file_name.split('.gif')[0].split('_')[-2]  # 假设分数总是倒数第二个元素
                score = float(score_str)
                total_score += score
            except ValueError:
                print(f"Warning: Score extraction failed for {file_name}")

        room_str = file_name.split('.gif')[0].split('_')[-1]
        room = float(room_str)
        total_room += room

    # 计算含有'succ'的文件比例
    succ_rate = num_succ / total_files if total_files > 0 else 0.0
    # 计算平均分数
    average_score = total_score / total_files if total_files > 0 else 0.0
    percentage_room = total_room / total_files if total_files > 0 else 0.0

    print(f"Total number of GIF files: {total_files}")
    print(f"Success rate: {succ_rate:.4f}")
    print(f"Average SEL: {average_score:.4f}")
    print(f"Percentage room visited: {percentage_room:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count GIF files and calculate success rate.")
    parser.add_argument("--dir_path", type=str, help="Directory path to search for GIF files")
    args = parser.parse_args()

    calculate_succ_rate_and_average_score_and_percentage_room(args.dir_path)

