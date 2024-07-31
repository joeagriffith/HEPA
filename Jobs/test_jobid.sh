#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=00:02:00

## Verify install:
python -c "import os;print(os.environ['PBS_JOBID'])"