#!/bin/bash
#SBATCH -J classification
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 5-00:00:00

#SBATCH -p normal
#SBATCH --gres=gpu:4
#SBATCH --nodelist=sist_gpu60
#SBATCH --cpus-per-task=4

#SBATCH --mail-type=all
#SBATCH --mail-user= """email"""
#SBATCH --output=./printlog/%j.out
#SBATCH --error=./printlog/%j.err


nvidia-smi

# 进入到项目目录
cd Powder_SingleCrystal_Liquid_ResNet18
# 执行训练脚本

/.../anaconda3/envs/fcn/bin/python3.7 train.py
