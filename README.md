# UNMamba: Cascaded Spatial-Spectral Mamba for Blind Hyperspectral Unmixing


[Dong Chen](https://github.com/Preston-Dong), [Junping Zhang](https://homepage.hit.edu.cn/zhangjunping), [Jiaxin Li](https://www.researchgate.net/profile/Li-Jiaxin-20)(李嘉鑫)

This paper has been accpeted by IEEE Geoscience and Remote Sensing Letters (GRSL) and can be downloaded [here](https://ieeexplore.ieee.org/document/10902420).

# $\color{red}{我的微信(WeChat): BatAug，欢迎交流与合作}$

### 我是李嘉鑫，25年毕业于中科院空天信息创新研究院的直博生，导师高连如研究员 ###

我的英文版本个人简历可在隔壁仓库下载，如您需要此简历模板可以通过微信联系我。
My english CV can be downloaded in this repository [![Static Badge](https://img.shields.io/badge/PDF-Download-blue])](https://github.com/JiaxinLiCAS/My-Curriculum-Vitae-CV-/blob/main/CV_JiaxinLi.pdf).

2025.09——, 就职于重庆邮电大学 计算机科学与技术学院 文峰副教授 $\color{red}{博后导师：韩军伟教授}$ 
【[官网](https://teacher.nwpu.edu.cn/hanjunwei.html)，[谷歌学术主页](https://scholar.google.com/citations?user=xrqsoesAAAAJ&hl=zh-CN&oi=ao)】

2020.09-2025.07 就读于中国科学院 空天信息创新研究院 五年制直博生 $\color{red}{导师：高连如研究员}$ 【[导师空天院官网](https://people.ucas.ac.cn/~gaolianru)，[谷歌学术主页](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN)】

2016.09-2020.7 就读于重庆大学 土木工程学院 测绘工程专业

From 2025.09, I work at the School of Computer Science and Technology (National Exemplary Software School), Chongqing University of Posts and Telecommunications, as a Wenfeng associate professor.
My postdoctoral supervisor is [Junwei Han](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN).

From 2020.09 to 2025.07, I am a PhD candidate at the Key Laboratory of Computational Optical Imaging Technology, Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China.
My supervisor is [Lianru Gao](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN).

From 2016.0 to 2020.7, I studied in the school of civil engineering at Chongqing University, Chongqing, China, for a Bachelor of Engineering.

# Getting Started

## Introduction
Blind hyperspectral unmixing (HU) has advanced significantly with the emergence of deep learning-based methods. However, the localized operations of convolutional neural networks (CNNs) and the high computational demands of Transformers present challenges for blind HU. This necessitates the development of image-level unmixing methods capable of capturing long-range spatial-spectral dependencies with low computational demands. This paper proposes a cascaded spatial-spectral Mamba model, termed UNMamba, which leverages the strengths of Mamba to efficiently model long-range spatial-spectral dependencies with linear computational complexity, achieving superior image-level unmixing performance with small parameters and operations.
Specifically, UNMamba first captures long-range spatial dependencies, followed by the extraction of global spectral features, forming long-range spatial-spectral dependencies, which are subsequently mapped into abundance maps. Then, the input image is reconstructed using the linear mixing model (LMM), incorporating weighted averages of multiple trainable random sequences and an endmember loss to learn endmembers. UNMamba is the first unmixing approach that introduces the state space models (SSMs). Extensive experimental results demonstrate that, without relying on any endmember initialization techniques (such as VCA), the proposed UNMamba achieves significantly high unmixing accuracy, outperforming state-of-the-art methods.
![Framework of the proposed UNMamba.](figs/framework.png)

## Results on Jasper Ridge
![Estimated abundances on the Jasper Ridge dataset.](figs/JR_abun_small_new.png)
(a)-(k) GT, FCLSU, SNMF-Net, DAEU, SIDAEU, MLAEM, DeepTrans, MLM-1DAE, A2SN, A2SAN and UNMamba.

## Installation

```bash
conda create -n UNMamba_env python=3.9
conda activate UNMamba_env
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install causal_conv1d
pip install mamba-ssm==1.2.0.post1
pip install spectral
pip install scikit-learn==1.4.1.post1
pip install calflops
pip install matplotlib
pip install torchsummary
```

## Training
```bash
python main_UNMambaLinear_JR.py
python main_UNMambaLinear_AP.py
python main_UNMambaLinear_UR.py
```


## Citation
If you find this project helpful for your research, please kindly consider citing this paper:
```bash
@ARTICLE{10902420,
  author={Chen, Dong and Zhang, Junping and Li, Jiaxin},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={UNMamba: Cascaded Spatial-Spectral Mamba for Blind Hyperspectral Unmixing}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Blind hyperspectral unmixing;linear mixing model;endmember loss;mamba;state space model},
  doi={10.1109/LGRS.2025.3545505}}
```
## Acknowledgements
Part of our UNMamba is referred to [MambaHSI](https://github.com/li-yapeng/MambaHSI) and [DeepTrans](https://github.com/preetam22n/DeepTrans-HSU). We thank all the contributors for open-sourcing.

