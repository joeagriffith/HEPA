#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=00:10:00

module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0

python -c "import time;print(f'loaded modules at: {time.time()}')"

source ~/ml-env/bin/activate

python -c "import time;print(f'activated env at: {time.time()}')"

cd ~/hepa/

python -c "import time;print(f'changed directory at: {time.time()}')"

python run.py test_mnist

python -c "import time;print(f'finished test_mnist at: {time.time()}')"
python run.py test_modelnet10
