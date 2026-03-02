# Improved ResUNet for Aluminum Alloy Microstructure Segmentation

This project provides an improved ResUNet deep learning model specifically designed for high-precision semantic segmentation of aluminum alloy microstructure images. This project is developed and extended based on the powerful open-source semantic segmentation framework, [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## 📖 Introduction

In the field of materials science, accurate quantification and analysis of microstructures are crucial for understanding material properties. Aiming at a custom dataset of aluminum alloy microstructure images, this project proposes an improved ResUNet model to achieve efficient and accurate segmentation of microstructures.

To verify the effectiveness of the model, this project also integrates various classic semantic segmentation baseline networks (such as UNet++, DeepLabV3+, SegNet, etc.) to facilitate comparative experiments.

## 📊 Dataset

This model was trained and evaluated on a custom aluminum alloy microstructure image dataset.

> Due to size limitations, the complete training dataset is not directly included in this repository. A small number of sample images are provided in the `aluminum_alloy_samples/` directory for reference.

## 🛠️ Installation

This project depends on PyTorch and MMSegmentation. Please follow the steps below to configure your environment:

Create and activate a virtual environment:

conda create -n mmseg python=3.8 -y
conda activate mmseg
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

Install MIM and MMCV:
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

## 🚀 Model Training and Inference
Use the trained weights to perform segmentation predictions on new aluminum alloy microstructure images:

(Note: You can add your command line instructions for training/inference here later)

## 🤝 Acknowledgments
We would like to thank the MMSegmentation team for providing an outstanding open-source semantic segmentation codebase. The architecture and configuration files of this project are built upon this framework.