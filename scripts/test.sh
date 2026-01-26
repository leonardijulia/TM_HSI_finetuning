#!/bin/bash
#SBATCH --account=IscrC_MOLCAGFM_0
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

set -e

module load python/3.11.7 
source /leonardo/home/userexternal/jleonard/experiments/JL_TT/bin/activate
export PYTHONPATH=$PYTHONPATH:/leonardo/home/userexternal/jleonard/experiments
cd /leonardo/home/userexternal/jleonard/experiments/ || exit 1

echo "Running test on $(hostname)"
echo "Started at $(date)"
nvidia-smi

EXPERIMENT=${1:-hyperview_1}
CKPT=${2:-outputs/hyperview_1/checkpoints/last.ckpt}

python src/run.py mode=test experiment=$EXPERIMENT ckpt_path=$CKPT

echo "Finished at $(date)"
#terratorch test -c configs/hyperview_1.yaml --ckpt_path outputs/hyperview/checkpoints/last.ckpt

