#!/bin/bash

# Create a virtual environment with Python 3.9 (here using conda):
conda create --name nlc python=3.9.7
conda activate nlc

# Install dependencies:
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt

# If you have any issues with ffmpeg, try:
# conda update ffmpeg
# pip install imageio-ffmpeg

# Optional. For library development, install developement dependencies.
pip install -r requirements-dev.txt
