#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=00:02:00

cd $PBS_O_WORKDIR

module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0
source ~/ml-env/bin/activate

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"
cd ~/hepa/
python run.py test_mnist_hpc
python run.py test_modelnet10_hpc