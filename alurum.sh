#! /bin/bash
#SBATCH --job-name=gpu_job                 # Job name
#SBATCH --time=02:00:00                    # Time limit (e.g., 2 hours)
#SBATCH --output=~/gpu_job.out             # Standard output and error log
#SBATCH --partition=Normal                 # Queue name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mem=32G                          # Memory per node
#SBATCH --gres=gpu:1                       # Request 1 GPU

  
salloc -p GPUA40 -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=64GB --time=03:00:00 --gres=gpu:1

ssh the provided node
then excute the above script:
    # module load cuda/cuda-11.8
    cd project/1030/conditioned
    source ~/.bashrc
    conda activate torch
    module load cuda

use "squeue -u zwlexa" to find available nodes
zgpuA403


nvm use node
npm install -g @anthropic-ai/claude-code
npm i -g @openai/codex

git pull origin main

git push origin work