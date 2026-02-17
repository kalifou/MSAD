#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --cluster=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=99000
#SBATCH --cpus-per-task=48

srun bash benchmark_windows_size_msad.sh 
