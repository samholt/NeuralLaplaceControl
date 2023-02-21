#!/bin/bash
conda create --name nlc python=3.9.7
conda activate nlc
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
pip install imageio
pip install pyvirtualdisplay
conda update ffmpeg
pip install TorchDiffEqPack
pip install imageio-ffmpeg
pip install torchlaplace
pip install -U scikit-learn scipy matplotlib
pip install pyglet==1.5.27
pip install seaborn