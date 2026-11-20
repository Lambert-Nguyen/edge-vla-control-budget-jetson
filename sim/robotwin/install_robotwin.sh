#!/usr/bin/env bash
# Install RoboTwin 2.0 on the RTX 4090 workstation, without sudo.
#
# The workstation has no conda, no C compiler and no ffmpeg, and we do not use sudo
# there. So everything goes into a Miniforge env in the home folder:
#   - Miniforge in ~/miniforge3, env "RoboTwin" with Python 3.10
#   - nvcc 12.1, gcc 12 and ffmpeg from conda, for pytorch3d, curobo and the demo videos
#   - RoboTwin in ~/robotwin/RoboTwin, assets from Hugging Face (about 16 GB)
#
#   bash sim/robotwin/install_robotwin.sh        # about 40 minutes
#
# Check: cd ~/robotwin/RoboTwin && bash collect_data.sh beat_block_hammer demo_clean 0
set -exo pipefail   # no -u: the conda compiler scripts use unset variables

mkdir -p ~/robotwin && cd ~/robotwin
if [ ! -x ~/miniforge3/bin/conda ]; then
  curl -fsSL -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3.sh -b -p ~/miniforge3
  rm Miniforge3.sh
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q "^RoboTwin " || conda create -y -n RoboTwin python=3.10
conda activate RoboTwin

conda install -y -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 ffmpeg
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
mkdir -p $CONDA_PREFIX/compiler-links
ln -sf $CC $CONDA_PREFIX/compiler-links/gcc
ln -sf $CXX $CONDA_PREFIX/compiler-links/g++
export PATH=$CONDA_PREFIX/compiler-links:$PATH
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=12

python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

[ -d RoboTwin ] || git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git submodule update --init --recursive XPolicyLab
bash scripts/_install.sh
bash scripts/_download_assets.sh
python -c "import sapien, mplib, pytorch3d, curobo, torch; print('RoboTwin env OK', torch.__version__)"
python scripts/test_render.py
