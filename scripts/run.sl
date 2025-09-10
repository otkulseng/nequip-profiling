#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4297


source .venv/bin/activate
module load cudatoolkit
srun nsys profile \
    --stats=true \
    -t nvtx,cuda \
    -o nsys_report \
    python src/nequip_profiling/benchmark_utils.py