#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task 1
#SBATCH -p gpu
#SBATCH --time=48:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=test
#SBATCH --output=test_job_%j.out

MODEL_ID=$1 #100
ALPHA_L_INF=$2 #0.005
ALPHA_L_1=$3 #2
NUM_ITER=$4 #10
LR=$5 #0.1
LR_MODE=$6

source ~/.bashrc
cd ~/scratch/projects/MSD/CIFAR10
python train.py -model_type wrn-34-10 -num_iter $NUM_ITER -epsilon_l_2 0 -epsilon_l_1 7.843 -epsilon_l_inf 0.0157 -alpha_l_1 $ALPHA_L_1 -alpha_l_inf $ALPHA_L_INF -lr_max $LR -model_id $MODEL_ID -lr_mode $LR_MODE -batch_size 200
