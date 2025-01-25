#!/bin/bash
#SBATCH --job-name="training_semantic_transformer"
#SBATCH --output=gpu_job.out
#SBATCH --error=gpu_job.err
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=50
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pqb.work@gmail.com
#SBATCH --time=2-00:00:00

module purge
module load Anaconda3
source activate 
conda activate pbao_semantictransformer
pip install -r requirement.txt
python train_large-10th-epoch.py --num_blocks 24 --epochs 10 --val_per_epoch 5 --seq_len 5 --batch_size 32