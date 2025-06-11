# [CVPR 2025] DiscoVLA: Discrepancy Reduction in Vision, Language, and Alignment for Parameter-Efficient Video-Text Retrieval

The official implementation of [DiscoVLA](https://arxiv.org/abs/2506.08887).

## Dataset
Prepare the dataset by following the instructions from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/).

For MSRVTT, the official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset).

For the convenience, the splits and captions can be found in sharing from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/),

```shell
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in sharing from [Frozen in Time](https://github.com/m-bain/frozen-in-time), i.e.,

```shell
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

## Train 

Please download the Pseudo Image Captions of MSRVTT from [here](https://pan.baidu.com/s/1c4MOV6_XZVKn_Cu79ZyvFw?pwd=py7q). For more details, please refer to our paper.

We conduct experiments on 4 A100x40G GPUs on MSRVTT. To set up the environment and run the experiments, execute the following commands:

```shell
bash scripts/create_env.sh
bash scripts/MSRVTT.sh
```

## Acknowledgement

This project builds upon the following open-source works: **[DRL](https://github.com/foolwood/DRL)**.

## Citation

```bibtex
@inproceedings{shen2025discovla,
  title={DiscoVLA: Discrepancy Reduction in Vision, Language, and Alignment for Parameter-Efficient Video-Text Retrieval},
  author={Shen, Leqi and Gong, Guoqiang and Hao, Tianxiang and He, Tao and Zhang, Yifeng and Liu, Pengzhang and Zhao, Sicheng and Han, Jungong and Ding, Guiguang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19702--19712},
  year={2025}
}
```