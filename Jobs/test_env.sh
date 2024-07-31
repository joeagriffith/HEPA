#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=00:02:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate  pytorch_env

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"