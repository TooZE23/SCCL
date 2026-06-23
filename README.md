# Unsupervised Domain Adaptive Object Detection via Semantic Consistency and Compactness Learning


Official PyTorch implementation of "Unsupervised Domain Adaptive Object Detection via Semantic Consistency and Compactness Learning"
Yajing Liu; Zhen Zhang; Yiming Su; Chunhui Hao; Xiyao Liu; Jiandong Tian

IEEE Transactions on Image Processing

## Overview
In this paper, we propose SCCL, a method for improving cross-domain holistic feature consistency and category feature discriminability for unsupervised domain adaptive object detection task.

## Dataset Preparation

| 数据集 | 任务 | 下载链接 |
|--------|------|----------|
| Cityscapes | Weather/S2R/Camera | [官网](https://www.cityscapes-dataset.com/) |
| Foggy Cityscapes | Weather | [官网](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) |
| Pascal VOC | Artistic |  [官网](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/) |
| Clipart | Artistic |  [官网](https://naoto0804.github.io/cross_domain_detection/) |
| Sim10K | S2R |  [官网](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) |
| KITTI| Camera |  [官网](https://www.cvlibs.net/datasets/kitti/) |

## 训练

```bash
python train_net_sccl.py \
    --num-gpus 1 \
    --config configs/faster_rcnn_VGG_cross_city.yaml 
```

## 推理与评估

```bash
python train_net_sccl.py \
    --num-gpus 1 \
    --config configs/faster_rcnn_VGG_cross_city.yaml 
    --eval-only 
    MODEL.WEIGHTS path/to/model_final.pth
```

## Citation

If you use SCCL in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```BibTeX
@article{liu2026unsupervised,
  author={Liu, Yajing and Zhang, Zhen and Su, Yiming and Hao, Chunhui and Liu, Xiyao and Tian, Jiandong},
  journal={IEEE Transactions on Image Processing}, 
  title={Unsupervised Domain Adaptive Object Detection via Semantic Consistency and Compactness Learning}, 
  year={2026},
  volume={35},
  number={},
  pages={2276-2291},
  }

```

