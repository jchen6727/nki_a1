#!/bin/bash
#SBATCH --job-name=loadCineca
#SBATCH -A icei_H_King
#SBATCH --partition=g100_usr_prod
#SBATCH --qos=g100_qos_dbg
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o testDipole.run
#SBATCH -e testDipole.err
#SBATCH --mail-user=erica.griffith@downstate.edu
#SBATCH --mail-type=end

source ~/.bashrc

cd /g100/home/userexternal/egriffit/A1/analysis/

srun python testDipole.py

wait