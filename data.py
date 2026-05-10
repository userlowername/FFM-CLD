import os
import shutil

# 源文件夹路径
source_dir = '/media/vipsl04/Harddisk/DFDC/test/frames'

# 目标文件夹路径
target_dir = '/media/vipsl04/Harddisk/DFDC/test1/fake'

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

# 初始化全局图片计数器
image_counter = 1

# 获取所有子文件夹（如 0001, 0002, 0003 等）
image_folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

# 遍历每个子文件夹
for image_folder in image_folders:
    image_folder_path = os.path.join(source_dir, image_folder)

    # 获取该子文件夹中的所有图片文件
    images = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

    # 如果图片数量不满16张，跳过该文件夹
    if len(images) < 16:
        print(f"跳过文件夹 {image_folder}，图片数量不足16张")
        continue

    # 只处理前16张图片
    for image in images[:16]:
        image_path = os.path.join(image_folder_path, image)

        # 目标文件路径，重新命名为连续编号：1.png, 2.png, 3.png, ...
        target_image_path = os.path.join(target_dir, f"{image_counter}.png")

        # 复制并重命名图片
        shutil.copy(image_path, target_image_path)
        print(f"复制 {image_path} 到 {target_image_path}")

        # 更新图片计数器
        image_counter += 1

print("完成复制和重命名操作")