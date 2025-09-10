#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4297


source .venv/bin/activate

srun python -m nequip_profiling.bencmark_utils.py