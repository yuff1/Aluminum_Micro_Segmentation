import os
import cv2
import numpy as np

# 设置JPG文件夹路径
images_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/crop'
Benign_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/Benign'
Malignant_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/Malignant'

# 获取JPG文件夹中所有文件的列表
images_files = os.listdir(images_folder)

# 遍历JPG文件夹中的所有文件
for file_name in images_files:
    image = cv2.imread(os.path.join(images_folder, file_name))
    if image.shape[0] < 100 and image.shape[1] < 100:
        os.system(f'cp {os.path.join(images_folder, file_name)} {os.path.join(Benign_folder, file_name)}')
    else:
        os.system(f'cp {os.path.join(images_folder, file_name)} {os.path.join(Malignant_folder, file_name)}')



