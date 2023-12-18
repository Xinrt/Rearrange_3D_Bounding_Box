#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=move_label_yolo
#SBATCH --output=/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/move_yolo_output.log
#SBATCH --error=/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/move_yolo_error.log
#SBATCH --time=10:00:00
#SBATCH --mem=64GB


# Extending PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/scratch/xt2191/ai2thor-rearrangement
export PYTHONPATH=$PYTHONPATH:/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box

conda init
source activate thor-rearrange

cd /scratch/xt2191/luyi/Rearrange_3D_Bounding_Box
cd scripts

echo "Start running"

python move_yolo_labels_with_rgb.py