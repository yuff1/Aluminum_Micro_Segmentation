import cv2
import os

# 输入文件夹和输出文件夹的路径
input_folder_path = "/home/zhangzifan/MaintoCode/2023-12-29-01/datasets/pred/swinunet"
output_folder_path = "/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/swinunet_crop"
images_folder_path = "/home/zhangzifan/MaintoCode/2023-12-29-01/datasets/test/images"


file_list1 = os.listdir("/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/Benign")
file_list2 = os.listdir("/home/zhangzifan/MaintoCode/2023-12-29-01/datasets_cls/test/Malignant")
for (index, file) in enumerate(file_list1):
    file_list1[index] = file.replace('cropped_', '')
for (index, file) in enumerate(file_list2):
    file_list2[index] = file.replace('cropped_', '')
mask_files = file_list1 + file_list2


# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 获取输入文件夹中所有mask文件的路径
# mask_files = [f for f in os.listdir(input_folder_path) if f.endswith('.png') or f.endswith('.jpg')]
# print(mask_files)

# 遍历每个mask文件
for mask_file in mask_files:
    # 读取mask图像
    mask_path = os.path.join(input_folder_path, mask_file)
    image_path = os.path.join(images_folder_path, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    # 寻找mask的外接矩形
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 获取外接矩形
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop出外接区域
        cropped_mask = mask[y:y+h, x:x+w]
        cropped_image = image[y:y+h, x:x+w]

        # 保存到新的文件夹
        output_image_path = os.path.join(output_folder_path, f"cropped_{mask_file}")
        cv2.imwrite(output_image_path, cropped_image)

print("处理完成！")
