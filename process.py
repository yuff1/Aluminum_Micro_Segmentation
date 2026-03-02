from PIL import Image
import os
from tqdm import tqdm

# 文件夹路径
folder_path = '/home/zhangzifan/MaintoCode/2023-4-29-01/data/label'

# 存储像素值的集合
pixels = set()

# 循环遍历文件夹中的每个文件
for filename in tqdm(os.listdir(folder_path)):
    # 检查文件是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 打开图片文件
        img = Image.open(os.path.join(folder_path, filename))
        # 获取像素数据
        img_pixels = img.getdata()
        # 将像素值添加到集合中
        pixels.update(img_pixels)

# 输出像素值的数量和列表
print('Number of pixel types:', len(pixels))
print('Pixel types:', pixels)
