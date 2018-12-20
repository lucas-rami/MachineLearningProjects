#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:0:0
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

echo STARTING AT `date`
python3 UNET_patch_120_rot.py
echo FINISHED at `date`
