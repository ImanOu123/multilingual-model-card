#!/bin/bash
#SBATCH --job-name=code
#SBATCH --partition=long
#SBATCH --output=vscode.out
#SBATCH --error=vscode.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=7-00:00:00
#SBATCH --mem=40G

source ~/.bashrc
conda activate new
jupyter notebook

