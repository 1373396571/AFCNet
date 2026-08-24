# AFCNet
## Introduction
We proposes a lightweight and efficient asymmetric feature complementary network (AFCNet), which can adaptively integrate the advantageous features from visible light and infrared modalities. Specifically, we propose a Frequency-Spatial Encoder(FSE) that first constructs a frequency filter in the frequency domain using a Butterworth function to remove noise from infrared and low-light images. Then, in the spatial domain, a set of depthwise separable convolutions is used to extract multi-receptive field features, eliminating the denoising impact and enhancing target features. Next, we use an asymmetric architecture based on Detail-Semantic Dual Attention (DSDA) to fuse infrared and visible light features, making full use of advantageous features and removing redundancy. Moreover, our asymmetric architecture significantly reduces the computational load. Additionally, our Long-range Detail Compensation (LDC) uses residual connections to inject shallow details into high-level features before the detection head, thereby improving pixel-level detection capability.
## Environment
pytorch 2.6.0

torchvision 0.21.0

python 3.11.9

cuda 12.0

We have also provided a requirements document so that everyone can reproduce these results.
## Datasets
The DroneVehicle dataset are available at https://github.com/VisDrone/DroneVehicle (accessed on 10 October 2022).

The LLVIP dataset are available at https://github.com/bupt-ai-cz/LLVIP (accessed in 17 October 2021).


<details>
<summary>📁 File structure</summary>

- Your dataset
  - train
    - rgb
      - images
      - labels
    - ir
      - images
      - labels
  - val
    - rgb
      - images
      - labels
    - ir
      - images
      - labels

</details>



