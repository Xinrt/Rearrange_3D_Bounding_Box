#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_yolov8n
#SBATCH --output=/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/train_yolov8n_sbatch_output.log
#SBATCH --error=/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/train_yolov8n_sbatch_error.log
#SBATCH --time=48:00:00
#SBATCH --mem=64GB


# Extending PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/scratch/xt2191/ai2thor-rearrangement
export PYTHONPATH=$PYTHONPATH:/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box

conda init
source activate thor-rearrange

cd /scratch/xt2191/luyi/Rearrange_3D_Bounding_Box
cd scripts

echo "Start running"

python train_yolo.py