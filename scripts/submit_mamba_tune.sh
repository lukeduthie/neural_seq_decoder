#!/bin/bash
#
#SBATCH --job-name=train_speech
#SBATCH --time=23:59:59
#SBATCH -p swl1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1 # number of gpus per task
#SBATCH --mail-type=END
#SBATCH --mail-user=dzoltow@stanford.edu
#SBATCH --mem=64GB

conda activate pytorch-speech

ml gcc/10.1.0
ml load cudnn/8.6.0.163
ml load cuda/11.7.1

NUM=50
SWEEPID="kounga-speech-bci/Speech_BCI_Tune/avah0voc"

srun -N 1 -n 1 -o tune_mamba.out wandb agent --count $NUM $SWEEPID