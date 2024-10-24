# MMI-Det: Exploring Multi-Modal Integration for Visible and Infrared Object Detection


This repository is an official implementation of the paper [MMI-Det: Exploring Multi-Modal Integration for Visible and Infrared Object Detection](https://ieeexplore.ieee.org/abstract/document/10570450).



## Introduction

The Visible-Infrared (VIS-IR) object detection is a challenging detection task, which combines visible and infrared data to give information on the category and location of objects in the scene. Therefore, the core of this task is to combine complementary information in the visible and infrared modalities to provide more object detection results for detection. The existing methods mainly face the problem of insufficient ability to perceive and combine visible-infrared modal information and have difficulty in balancing the optimization directions of the fusion and detection tasks. To solve these problem, we propose the MMI-Det which is a multi-modal fusion method for visible and infrared object detection. The method can provide a good combination of complementary information in the visible-infrared modalities and output accurate and robust object information. Specifically, to improve the ability of the model to perceive environment at the visible-infrared image level, we designed the Contour Enhancement Module. Furthermore, to extract complementary information from VIS and IR modalities, we design the Fusion Focus Module. It can extract different frequency spectral features of the visible and infrared modalities and focus on the key information of the object at different spatial locations. Moreover, we design the Contrast Bridge Module to improve the ability to extract modal invariant features in the visible-infrared scene. Finally, to ensure that our model can balance the optimization directions of image fusion and object detection, we design the Info Guided Module as a way to improve the effectiveness of the model's training optimization. We implement extensive experiments on the public FLIR, M3FD, LLVIP, TNO and MSRS datasets, and compared with previous methods, our method achieves better performance with powerful multi-modal information perception capabilities.

<br/>
<div align="center">
  <img src="./framework.png" width="90%"/>

  Fig. 1: Overall architecture of the proposed MMI-Det model.
</div>

## Model
The pretrained model in the M3FD dataset can be downloaded on 
https://drive.google.com/file/d/1zdQTcEwpzmns-4fQ8OvZ6MghW1V-feMB/view?usp=drive_link


## Citing MMI-Det
If you find MMI-Det useful in your research, please consider citing:
```bibtex
@ARTICLE{10570450,
  author={Zeng, Yuqiao and Liang, Tengfei and Jin†, Yi and Li, Yidong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={MMI-Det: Exploring Multi-Modal Integration for Visible and Infrared Object Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Image fusion;Object detection;Task analysis;Optimization;Circuits and systems;Data mining;multi-spectral object detection;multi-modal integration;image fusion;Fourier transformation},
  doi={10.1109/TCSVT.2024.3418965}}
```


## License

This repository is released under the Apache 2.0 license. Please see the [LICENSE](./LICENSE) file for more information.
