import cv2
import numpy as np
import mmcv
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import matplotlib.pyplot as plt
import time  # 引入时间模块

# Load configuration file
cfg = mmcv.Config.fromfile('F:/fyf/ResUnet_all/seg_code/my_config/unet.py')

# Build the segmentor
model = init_segmentor(cfg, 'F:/fyf/ResUnet_all/seg_code/work_dirs/unet 10-5/iter_100000.pth', device='cuda:0')

# Test a single image 
img = 'sem4-32.png'  # replace 'test.jpg' with your image file

start_time = time.time()  # 记录开始时间
result = inference_segmentor(model, img)
end_time = time.time()  # 记录结束时间

# 打印出所用时间
print("Segmentation time: {:.2f} seconds".format(end_time - start_time))

# Convert result to numpy array
result = np.array(result)

# Create a palette with two colors
palette = [[0, 0, 0],  # Black color
           [255, 255, 255]]  # White color
palette = np.array(palette, dtype=np.uint8)

# Create a new image based on the segmentation result
new_img = np.zeros((result.shape[1], result.shape[2], 3), dtype=np.uint8)
for i in range(palette.shape[0]):
    new_img[result[0]==i] = palette[i]

# Display the new image
plt.imshow(new_img)
plt.axis('off')
plt.show()
print(type(new_img))
# Save the new image
cv2.imwrite('new-32.png', new_img)