#!/bin/bash

conda create -n discovla python=3.10 -y

source activate discovla
conda activate discovla

pip install numpy==1.23.0
pip install decord
pip install pandas
pip install ftfy
pip install regex
pip install tqdm
pip install opencv-python
pip install functional
pip install torch==2.2.0 torchvision==0.17.0
pip install timm==0.4.12
pip install --upgrade --force-reinstall scipy
pip install numpy==1.23.0
pip install einops==0.7.0