#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/scratch/ys3426/result/%j_%x_pooler.out
#SBATCH --error=/scratch/ys3426/result/%j_%x_pooler.err
#SBATCH --account=ds_ga_1012-2023sp
#SBATCH --open-mode=append
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -c 4
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3426@nyu.edu


singularity exec --nv \
--bind /scratch/ys3426 --overlay /scratch/ys3426/overlay-25GB-500K.ext3:ro \
/scratch/ys3426/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash -c "source /ext3/env.sh; python FL_pooler.py"
