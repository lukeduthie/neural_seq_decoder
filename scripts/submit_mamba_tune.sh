#!/bin/bash
#
#SBATCH --job-name=tune_mamba
#SBATCH --time=23:59:59
#SBATCH -p swl1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1 # number of gpus per task
#SBATCH --mail-type=END
#SBATCH --mail-user=mkounga@stanford.edu
#SBATCH --mem=64GB

conda activate pytorch-speech

ml gcc/10.1.0
ml load cudnn/8.6.0.163
ml load cuda/11.7.1

NUM=100
SWEEPID="kounga-speech-bci/Mamba_Tune/o9p9564e"

srun -N 1 -n 1 -o tune_mamba.out wandb agent --count $NUM $SWEEPID