#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=01:00:00

module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0

python -c "import time;msg='loaded modules at:'+time.strftime('%H:%M:%S',time.localtime());print(msg)"

source ~/ml-env/bin/activate

python -c "import time;msg='activated env at:'+time.strftime('%H:%M:%S',time.localtime());print(msg)"

cd ~/hepa/

python -c "import time;msg='changed directory at:'+time.strftime('%H:%M:%S',time.localtime());print(msg)"

python run.py test_mnist
python -c "import time;msg='finished test_mnist at:'+time.strftime('%H:%M:%S',time.localtime());print(msg)"

python run.py test_modelnet10
python -c "import time;msg='finished test_modelnet10 at:'+time.strftime('%H:%M:%S',time.localtime());print(msg)"