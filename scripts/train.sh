#!/bin/bash
#SBATCH --account=pMI25_DICA_0
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=7:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

set -e

module load python/3.11.7
source /leonardo/home/userexternal/jleonard/ICLR_exp/ICLR_TM/bin/activate
export PYTHONPATH=$PYTHONPATH:/leonardo/home/userexternal/jleonard/ICLR_exp
cd /leonardo/home/userexternal/jleonard/ICLR_exp/ || exit 1

echo "Running on $(hostname)"
echo "Started at $(date)"
nvidia-smi

EXPERIMENT=${1:-hyperview_1} # Default to hyperview_1 if not provided
EXTRA_ARGS="${@:2}"          # Any additional Hydra overrides

python src/run.py mode=train experiment=$EXPERIMENT $EXTRA_ARGS

echo "Finished at $(date)"
#terratorch fit -c configs/enmap_cdl.yaml
