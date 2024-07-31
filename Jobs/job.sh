#!/bin/bash
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=00:30:00

module purge
module load anaconda3/personal
conda create -n pytorch_env -c conda-forge cudatoolkit=11.8 python=3.11
conda activate pytorch_env
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install numpy
conda install anaconda::pandas
conda install anaconda::pillow
conda install conda-forge::pyyaml
conda install anaconda::scipy
conda install pytorch::torchvision
conda install conda-forge::tqdm

python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

cd ${PBS_O_WORKDIR}
python3 run.py test_mnist_hpc
python3 run.py test_modelnet_hpc