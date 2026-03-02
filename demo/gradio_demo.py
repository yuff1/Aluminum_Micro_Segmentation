import cv2
import numpy as np
import gradio as gr
# import torch
import albumentations as Alb
import os
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
import numpy as np
import mmcv
import sys

sys.path.append("/data2/zhangzifan/code_dir/2023-12-29-01")
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms
from PIL import Image



def generate(src):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    cv2.imwrite("./tmp/A.jpg", src)
    result = inference_segmentor(seg_model, "./tmp/A.jpg")
    mask = result[0]

    image_cls = Image.open("./tmp/A.jpg")
    image_cls = image_cls.convert('RGB')
    image_draw = image_cls.copy()
    image_cls = transform(image_cls)
    image_cls = torch.reshape(image_cls, (1, 3, 224, 224))  # 修改待预测图片尺寸，需要与训练时一致
    image_cls = image_cls.cuda()
    print(image_cls.shape)
    cls_model.eval()
    with torch.no_grad():
        output = cls_model(image_cls)
    # print(int(output.argmax(1)))
    # 对结果进行处理，使直接显示出预测的种类
    draw = ImageDraw.Draw(image_draw)
    font = ImageFont.load_default()  # 选择字体
    draw.text((10, 10), f"class: {data_class[int(output.argmax(0))]}", (255, 255, 255), font=font)  # 在图像上指定位置绘制文字
    annotation_masks = [np.array(image_draw), mask * 255]
    return annotation_masks


if __name__ == "__main__":
    seg_model = init_segmentor('/home/zhangzifan/MaintoCode/2023-12-29-01/work_dirs/transunet_new/transunet_new.py',
                               '/home/zhangzifan/MaintoCode/2023-12-29-01/work_dirs/transunet_new/iter_25000.pth', device='cuda:0')
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    cls_model = torch.load(
        "/home/zhangzifan/MaintoCode/2023-12-29-02/checkpoint/hierarchy_swin_transformer/hierarchy_swin_transformer_best.pth",
        map_location=torch.device("cuda:0"))
    data_class = ['Benign', 'Malignant']

    with gr.Blocks() as application:
        gr.Markdown(value="Gradio Demo")
        with gr.Tab("Gradio Demo"):
            with gr.Row():
                with gr.Row():
                    with gr.Accordion("upload"):
                        src_image = gr.Image(
                            source='upload',
                            label='原图',
                            elem_id='image',
                            brush_radius=20,
                        )

                    with gr.Accordion("settings"):
                        with gr.Column():
                            with gr.Row():
                                submit = gr.Button("submit")
                            with gr.Column():
                                annotation_masks = gr.Gallery(label='map list')
                                submit.click(generate,
                                             inputs=[src_image],
                                             outputs=[annotation_masks],
                                             )

    # 应用队列设置
    application.queue(
        concurrency_count=1,
        api_open=True
    )

    # 应用启动设置
    application.launch(
        server_name='10.1.20.107',
        server_port=6801,
        debug=True,
        show_api=True
    )


