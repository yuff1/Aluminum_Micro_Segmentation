import cv2
import numpy as np
import os
import random

# 输入文件夹和输出文件夹
image_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/swinunet_crop'
other_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/swinunet_test'
Benign_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/Benign'
Malignant_folder = '/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/Malignant'


image_files = os.listdir(Benign_folder)
for image_file in image_files:
    # 读取图像和相应的mask
    os.system(f'cp {os.path.join(image_folder, image_file)} {os.path.join(other_folder, "Benign", image_file)}')


image_files = os.listdir(Malignant_folder)
for image_file in image_files:
    # 读取图像和相应的mask
    os.system(f'cp {os.path.join(image_folder, image_file)} {os.path.join(other_folder, "Malignant", image_file)}')



