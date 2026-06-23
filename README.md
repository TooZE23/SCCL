# Unsupervised Domain Adaptive Object Detection via Semantic Consistency and Compactness Learning


Official PyTorch implementation of "Unsupervised Domain Adaptive Object Detection via Semantic Consistency and Compactness Learning"
Yajing Liu; Zhen Zhang; Yiming Su; Chunhui Hao; Xiyao Liu; Jiandong Tian

IEEE Transactions on Image Processing

## Overview
In this paper, we propose SCCL, a method for improving cross-domain holistic feature consistency and category feature discriminability for unsupervised domain adaptive object detection task.

## Dataset Preparation

| Dataset | Task | Download |
|--------|------|----------|
| Cityscapes | Weather/S2R/Camera | [Link](https://www.cityscapes-dataset.com/) |
| Foggy Cityscapes | Weather | [Link](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) |
| Pascal VOC | Artistic |  [Link](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/) |
| Clipart | Artistic |  [Link](https://naoto0804.github.io/cross_domain_detection/) |
| Sim10K | S2R |  [Link](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) |
| KITTI| Camera |  [Link](https://www.cvlibs.net/datasets/kitti/) |

## Train

```bash
python train_net_sccl.py \
    --num-gpus 1 \
    --config configs/faster_rcnn_VGG_cross_city.yaml 
```

## Evaluate

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

