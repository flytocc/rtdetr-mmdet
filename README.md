### Feel free to open an issue

## TODO

- [ ] RTDETRv4
- [ ] Instance Segmentation for DEIM / DEIMv2
- [x] Instance Segmentation for RTDETR / RTDETRv2
- [x] DEIMv2

***

## Training Logs

DEIM v2                         | ours | official | gap    | checked | log
:-------------------------------|:-----|:---------|:-------|:--------|:----
deimv2_dinov3_m_8xb4-102e_coco  | 53.1 | 53.0     | `+0.1` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/deimv2_dinov3_m_8xb4-102e_coco.log)
deimv2_dinov3_l_8xb4-68e_coco   | 56.0 | 56.0     | `+0.0` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/deimv2_dinov3_l_8xb4-68e_coco.log)
deimv2_dinov3_x_8xb4-58e_coco   | 57.8 | 57.8     | `+0.0` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/deimv2_dinov3_x_8xb4-58e_coco.log)

DEIM                            | ours | official | gap    | checked | log
:-------------------------------|:-----|:---------|:-------|:--------|:----
deim_hgnetv2_n_8xb16-160e_coco  | 42.9 | 43.0     | `-0.1` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/deim_hgnetv2_n_8xb16-160e_coco.log)
deim_r18vd_8xb2-120e_coco       | 49.2 | 49.0     | `+0.2` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/deim_r18vd_8xb2-120e_coco.log)

D-FINE                          | ours | official | gap    | checked | log
:-------------------------------|:-----|:---------|:-------|:--------|:----
dfine_hgnetv2_n_8xb16-160e_coco | 42.6 | 42.8     | `-0.2` |         | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/dfine_hgnetv2_n_8xb16-160e_coco.log)

RT-DETR v2                       | ours | official | gap    | checked | log
:--------------------------------|:-----|:---------|:-------|:--------|:----
rtdetrv2_r18vd_8xb2-120e_coco    | 48.3 | 48.1     | `+0.2` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/rtdetrv2_r18vd_1xb16-120e_coco.log)
rtdetrv2_r34vd_dsp_8xb2-12e_coco | 49.3 | 49.1     | `+0.2` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/rtdetrv2_r34vd_dsp_8xb2-12e_coco.log)

RT-DETR                         | ours | official | gap    | checked | log
:-------------------------------|:-----|:---------|:-------|:--------|:----
rtdetr_r18vd_8xb2-72e_coco      | 46.7 | 46.5     | `+0.2` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/rtdetr_r18vd_1xb16-72e_coco.log)
rtdetr_r50vd_8xb2-72e_coco      | 53.1 | 53.1     | `+0.0` | ✅      | [download](https://github.com/flytocc/rtdetr-mmdet/releases/download/logs/rtdetr_r50vd_1xb16-72e_coco.log)

***

## Citation

> [**DETRs Beat YOLOs on Real-time Object Detection**](https://arxiv.org/abs/2304.08069) [RT-DETR] [CVPR 2024] <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/lyuwenyu/RT-DETR)  [![arXiv](https://img.shields.io/badge/Arxiv-2304.08069-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2304.08069) <br>

```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

> [**RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer**](https://arxiv.org/abs/2407.17140) <br>
> Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/lyuwenyu/RT-DETR)  [![arXiv](https://img.shields.io/badge/Arxiv-2407.17140-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.17140) <br>

```
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```

> [**D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement**](https://arxiv.org/abs/2410.13842) [ICLR 2025 Spotlight] <br>
> Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, Feng Wu <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Peterande/D-FINE)  [![arXiv](https://img.shields.io/badge/Arxiv-2410.13842-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.13842) <br>

```
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

> [**DEIM: DETR with Improved Matching for Fast Convergence**](https://arxiv.org/abs/2412.04234) [CVPR 2025] <br>
> Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, Xi Shen <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Intellindust-AI-Lab/DEIM)  [![arXiv](https://img.shields.io/badge/Arxiv-2412.04234-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.04234) <br>

```
@misc{huang2024deim,
      title={DEIM: DETR with Improved Matching for Fast Convergence},
      author={Shihua, Huang and Zhichao, Lu and Xiaodong, Cun and Yongjun, Yu and Xiao, Zhou and Xi, Shen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```

> [**Real-Time Object Detection Meets DINOv3**](https://arxiv.org/abs/2509.20787) [DEIMv2] <br>
> Shihua Huang, Yongjie Hou, Longfei Liu, Xuanlong Yu, Xi Shen <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Intellindust-AI-Lab/DEIMv2)  [![arXiv](https://img.shields.io/badge/Arxiv-2509.20787-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.20787) <br>

```
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}
```

> [**RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models**](https://arxiv.org/abs/2510.25257) <br>
> Zijun Liao<sup>* </sup>, Yian Zhao<sup>* </sup>, Xin Shan, Yu Yan, Chang Liu, Lei Lu, Xiangyang Ji, Jie Chen <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RT-DETRs/RT-DETRv4)  [![arXiv](https://img.shields.io/badge/Arxiv-2510.25257-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.25257) <br>

```
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Zijun Liao and Yian Zhao and Xin Shan and Yu Yan and Chang Liu and Lei Lu and Xiangyang Ji and Jie Chen},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```
