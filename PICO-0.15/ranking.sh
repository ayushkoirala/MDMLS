#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16			    # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1		    # Specify tasks per node
#SBATCH -t 24:00:00               # Specify maximum time limit (hour: minute: second)
# SBATCH -A MedVQA               	# Specify project name
# SBATCH -J get_memery               # Specify job names


echo "Start Ranking job"
echo "Loading miniconda module and activate envs"
module load Miniconda3/22.11.1-1 
conda activate /home/akoirala/Thesis/envs        

echo "===================================================================== Running python file ======================================================================"
python3 ranking.py \
    --log \


echo "===================================================================== Finished python file ====================================================================="