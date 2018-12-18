#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

echo STARTING AT `date`
python3
echo FINISHED at `date`
