#! /bin/bash
#SBATCH --account=IscrC_MOLCAGFM_0
#SBATCH --output=Pred_%j.out
#SBATCH --error=Pred_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

set -e

# Activate environment
module load python/3.11.7
source /leonardo/home/userexternal/jleonard/experiments/TM/bin/activate

# Ensure Python can see the src/ folder
export PYTHONPATH=$PYTHONPATH:/leonardo/home/userexternal/jleonard/experiments

# Enter project
cd /leonardo/home/userexternal/jleonard/experiments/ || exit 1

echo "Running test on $(hostname)"
echo "Started at $(date)"
nvidia-smi

python configs/predict_enmap_bnetd.py

echo "Prediction finished at $(date)"
