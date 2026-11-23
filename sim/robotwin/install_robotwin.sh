#!/usr/bin/env bash
# Install RoboTwin 2.0 on the RTX 4090 workstation, without sudo.
#
# The workstation has no conda, no C compiler and no ffmpeg, and we do not use sudo
# there. So everything goes into a Miniforge env in the home folder:
#   - Miniforge in ~/miniforge3, env "RoboTwin" with Python 3.10
#   - nvcc 12.1, gcc 12 and ffmpeg from conda, for pytorch3d, curobo and the demo videos
#   - RoboTwin at the commit in configs/robotwin_tasks.yaml (robotwin_commit), in
#     ~/robotwin/RoboTwin-<commit>, with assets from Hugging Face (about 16 GB)
#
#   bash sim/robotwin/install_robotwin.sh        # about 40 minutes
#
# Every setup has to run the same RoboTwin, so the script checks out the pinned
# commit, and stops if the folder holds a different one. To install another
# version, set ROBOTWIN_REF (a commit, tag or branch); ROBOTWIN_DIR sets the
# folder. The workstation's checkout of 13c3c47 is ~/robotwin/RoboTwin-0726:
#
#   ROBOTWIN_DIR=~/robotwin/RoboTwin-0726 bash sim/robotwin/install_robotwin.sh
#
# RoboTwin moved its scripts from script/ to scripts/ after 13c3c47 and added the
# XPolicyLab submodule; the script handles both layouts.
# The Hy-VLA policy adapter (policy/hy_vla) is not part of RoboTwin; it comes from
# the Hy-VLA repository (see export/robotwin_baseline.py).
#
# Check: cd ~/robotwin/RoboTwin-13c3c47 && bash collect_data.sh beat_block_hammer demo_clean 0
set -exo pipefail   # no -u: the conda compiler scripts use unset variables

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PINNED=$(sed -n 's/^robotwin_commit:[[:space:]]*\([0-9a-f]*\).*/\1/p' "$REPO_ROOT/configs/robotwin_tasks.yaml")
ROBOTWIN_REF=${ROBOTWIN_REF:-$PINNED}
[ -n "$ROBOTWIN_REF" ] || { echo "no robotwin_commit in configs/robotwin_tasks.yaml"; exit 1; }
ROBOTWIN_DIR=${ROBOTWIN_DIR:-$HOME/robotwin/RoboTwin-$ROBOTWIN_REF}

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

if [ -d "$ROBOTWIN_DIR" ]; then
  cd "$ROBOTWIN_DIR"
  want=$(git rev-parse --verify "$ROBOTWIN_REF^{commit}" 2>/dev/null || true)
  if [ -z "$want" ] || [ "$(git rev-parse HEAD)" != "$want" ]; then
    echo "$ROBOTWIN_DIR is at $(git rev-parse --short HEAD), not $ROBOTWIN_REF. Use another ROBOTWIN_DIR."
    exit 1
  fi
else
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_DIR"
  cd "$ROBOTWIN_DIR"
  git checkout --detach "$ROBOTWIN_REF"
fi
if [ -d scripts ]; then           # after 13c3c47
  S=scripts
  git submodule update --init --recursive XPolicyLab
else                              # 13c3c47 and before
  S=script
fi
bash $S/_install.sh
bash $S/_download_assets.sh
python -c "import sapien, mplib, pytorch3d, curobo, torch; print('RoboTwin env OK', torch.__version__)"
python $S/test_render.py
echo "RoboTwin $(git rev-parse --short HEAD) installed in $ROBOTWIN_DIR"
