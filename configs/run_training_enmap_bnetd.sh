#!/bin/bash
#SBATCH --account=IscrC_MOLCAGFM_0
#SBATCH --output=Job_%j.out
#SBATCH --error=Job_%j.err
#SBATCH --time=7:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

module load python/3.11.7--gcc--10.2.0
source /leonardo/home/userexternal/jleonard/experiments/JL_TT/bin/activate

cd /leonardo/home/userexternal/jleonard/experiments/ || exit 1

echo "Running on " `hostname`
echo "Working dir is /leonardo/home/userexternal/jleonard/experiments/"
echo "Job started at " `date`
nvidia-smi
export PYTHONPATH=$PYTHONPATH:/leonardo/home/userexternal/jleonard/experiments

terratorch fit -c configs/enmap_bnetd.yaml

echo "Job finished at " `date`