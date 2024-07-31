#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=00:30:00

module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0
source ~/ml-env/bin/activate

cd ~/hepa/
python run.py test_mnist
python run.py test_modelnet10