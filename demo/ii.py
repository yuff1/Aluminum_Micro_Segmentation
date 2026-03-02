
import cv2
import numpy as np
import mmcv
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import matplotlib.pyplot as plt

# Load configuration file
cfg = mmcv.Config.fromfile('F:/fyf/ResUnet_all/seg_code/my_config/unet.py')

# Build the segmentor
model = init_segmentor(cfg, 'F:/fyf/ResUnet_all/seg_code/work_dirs/unet/iter_100000.pth', device='cuda:0')

# Test a single image 
img = 'sem4-32.png'  # replace 'test.jpg' with your image file
result = inference_segmentor(model, img)

# Create a palette with two colors
palette = [[0, 0, 0],  # Black color
           [255, 255, 255]]  # White color
palette = np.array(palette, dtype=np.uint8)

# Visualize the result
show_result_pyplot(model, img, result, palette)
# print(type(img))
# Save the result
plt.savefig('32.png')