#!/bin/bash
#SBATCH --account=pMI25_DICA_0
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

set -e

module load python/3.11.7
source /leonardo/home/userexternal/jleonard/ICRL_exp/ICLR_TM/bin/activate
export PYTHONPATH=$PYTHONPATH:/leonardo/home/userexternal/jleonard/ICRL_exp
cd /leonardo/home/userexternal/jleonard/ICRL_exp || exit 1

echo "Running inference on $(hostname)"
echo "Started at $(date)"
nvidia-smi

EXPERIMENT=${1:-enmap_bnetd}
CKPT=${2:-outputs/enmap_bnetd/checkpoints/last.ckpt}

python src/run.py mode=predict experiment=$EXPERIMENT ckpt_path=$CKPT ## HERE ALSO POSSIBLE TO USE THE SPECIFIC SCRIPTS

echo "Finished at $(date)"