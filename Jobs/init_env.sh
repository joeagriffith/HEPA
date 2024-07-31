#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=00:15:00

module load anaconda3/personal
conda create -n pytorch_env -c conda-forge cudatoolkit=11.8 python=3.11
conda activate pytorch_env
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# For both TensorFlow and Pytorch, use the command in the next section.
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118