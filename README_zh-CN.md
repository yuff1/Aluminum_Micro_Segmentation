# Improved ResUNet for Aluminum Alloy Microstructure Segmentation

本项目提供了一种改进的 ResUNet 深度学习模型，专门用于铝合金微观图像的高精度语义分割。本项目基于强大的开源语义分割框架 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 进行开发和扩展。

## 📖 简介 (Introduction)

在材料科学领域，准确量化和分析微观结构对于理解材料性能至关重要。本项目针对自制的铝合金微观图像数据集，提出了一种改进的 ResUNet 模型，以实现对微观组织的高效、精准分割。

为了验证模型的有效性，本项目还集成了多种经典的语义分割基线网络（如 UNet++, DeepLabV3+, SegNet 等）以便于进行横向对比实验。

## 📊 数据集 (Dataset)

本模型在一个自制的铝合金微观图像数据集上进行了训练和评估。

> 由于体积限制，完整的训练数据集未直接包含在本仓库中。
在 `aluminum_alloy_samples/` 目录下提供了少量样本供参考。

## 🛠️ 安装与环境配置 (Installation)
本项目依赖于 PyTorch 和 MMSegmentation。请按照以下步骤配置您的环境：

创建并激活虚拟环境：

conda create -n mmseg python=3.8 -y
conda activate mmseg
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

安装 MIM 和 MMCV：
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

## 🚀 模型训练与预测 (Inference)
使用训练好的权重对新的铝合金微观图像进行分割预测：

## 🤝 致谢 (Acknowledgments)
感谢 MMSegmentation 团队提供的极其优秀的开源语义分割代码库，本项目的架构和配置文件均基于此框架构建。

