#!/bin/bash
conda create --name nlc python=3.9.7
conda activate nlc
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
conda update ffmpeg
pip install imageio-ffmpeg