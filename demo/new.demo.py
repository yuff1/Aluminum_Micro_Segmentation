# Copyright (c) OpenMMLab. All rights reserved.
import os
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot


# ---------------------- 在这里直接修改参数 ----------------------
MY_CONFIG = "./my_config/resunet_ca.py"          # 配置文件路径
CHECKPOINT = "./work_dirs/resunet_ca/iter_20000.pth"  # 模型权重路径
DEVICE = "cuda:0"                               # 运行设备（cuda:0 或 cpu）
IMAGE_PATH = "./tt/jx/answer1.png"                 # 单张图像的路径（直接修改这里）
SAVE_PATH = "./tt/jx2/"                         # 结果保存目录
OPACITY = 1.0                                   # 分割图透明度（0-1之间）
# 类别调色板（根据你的4类别修改，顺序对应ID 0-3）
PALETTE = [
    [0, 0, 0],          # 0: __background__（黑色）
    [255, 0, 0],        # 1: Assembled solid（红色）
    [0, 255, 0],        # 2: Bearing housing（绿色）
    [0, 0, 255]         # 3: Shaft（蓝色）
]
# ----------------------------------------------------------------


def main():
    # 检查输入图像是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：图像文件不存在 → {IMAGE_PATH}")
        return

    # 创建保存目录（不存在则自动创建）
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 初始化模型
    print(f"正在初始化模型...")
    seg_model = init_segmentor(MY_CONFIG, CHECKPOINT, device=DEVICE)
    print(f"模型初始化完成 → 设备：{DEVICE}")

    # 获取图像文件名（用于保存结果）
    image_filename = os.path.basename(IMAGE_PATH)  # 提取文件名（如test.jpg）
    save_file_path = os.path.join(SAVE_PATH, image_filename)  # 完整保存路径

    # 单张图像推理
    print(f"正在处理图像 → {IMAGE_PATH}")
    result = inference_segmentor(seg_model, IMAGE_PATH)

    # 保存分割结果
    show_result_pyplot(
        seg_model,
        IMAGE_PATH,
        result,
        PALETTE,
        opacity=OPACITY,
        out_file=save_file_path
    )

    print(f"处理完成！结果已保存至 → {save_file_path}")


if __name__ == '__main__':
    main()