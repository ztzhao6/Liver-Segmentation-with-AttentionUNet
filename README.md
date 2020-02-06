# Liver Segmentation with Attention-UNet
Automatic Liver Segmentation of CT Volumes Using 2D Attention-UNet

References：[Attention gated networks: Learning to leverage salient regions in medical images](https://arxiv.org/pdf/1808.08114.pdf "attention-unet")
* Automatic segmentation of the liver is an important step towards deriving quantitative biomarkers for accurate clinical diagnosis and
computer-aided decision support systems.
* Used the 2D U-Net architecture integrating attention mechanism and dice loss for accurate liver segmentation.
* Understood and processed the image data using Numpy and Scikit-Image. Implemented Attention-UNet using PyTorch.
* Achieved Dice scores over 96% for the liver in LiTS challenge dataset with computation times below 100s per volume. Applied the
technology to Nanjing Zhongda Hospital to help doctors in clinical research.
