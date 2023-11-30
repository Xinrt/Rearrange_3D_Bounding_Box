#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_generate_data
#SBATCH --output=/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/sbatch_output.log
#SBATCH --error=/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/sbatch_error.log
#SBATCH --time=40:00:00
#SBATCH --mem=50GB

# Setting up Vulkan SDK environment
# cd /scratch/xt2191/luyi/vulkan/1.2.189.0
# source setup-env.sh

# vulkaninfo

# Extending PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/scratch/xt2191/ai2thor-rearrangement
export PYTHONPATH=$PYTHONPATH:/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box

conda init
source activate thor-rearrange

cd /scratch/xt2191/luyi/Rearrange_3D_Bounding_Box
cd scripts

echo "Start running"

python generate_openness_dataset.py --logdir "/vast/xt2191/dataset"