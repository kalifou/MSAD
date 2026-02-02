#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --cluster=hpda2 
#SBATCH --partition=hpda2_compute
#SBATCH --time=05:00:00
#SBATCH --output=cpu-out.%j
#SBATCH --error=cpu-err.%j
#SBATCH --mem=99000
#SBATCH --cpus-per-task=48

srun bash create_windowed_benchmark_SPAICE.sh
