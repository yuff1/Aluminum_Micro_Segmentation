import os
import json
import numpy as np
from PIL import Image, ImageDraw

# 定义类别对应的颜色
class_colors = {
    "scratch": (100, 100, 100),
    "stain": (200, 200, 200),
    "edgeDamage": (255, 255, 255)
}

# 输入和输出文件夹路径
input_folder = './datasets/json'  # 替换为包含Labelme JSON文件的文件夹路径
output_folder = './datasets/mask'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 循环处理每个JSON文件
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        with open(os.path.join(input_folder, filename), 'r') as f:
            data = json.load(f)

        # 获取图像尺寸
        img_width = data['imageWidth']
        img_height = data['imageHeight']

        # 创建一个空白图像
        mask = Image.new('RGB', (img_width, img_height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)

        # 循环处理每个标注对象
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            polygon_points = [tuple(p) for p in points]
            draw.polygon(polygon_points, fill=class_colors[label])

        # 保存mask图像
        mask.save(os.path.join(output_folder, os.path.splitext(filename)[0] + '.png'))

print('转换完成')

