#!/bin/bash
#SBATCH -n 4                     # 4 cores CPU
#SBATCH -N 1                     # Misma máquina
#SBATCH -D /fhome/pmlai09/miguel/env/pacman
#SBATCH -t 7-00:00               # Tiempo máximo
#SBATCH -p tfg                   # Partición
#SBATCH --mem 8192               # 8GB RAM
#SBATCH -o %x_%u_%j.out          # STDOUT
#SBATCH -e %x_%u_%j.err          # STDERR
#SBATCH --gres=none

sleep 5

python /fhome/pmlai09/miguel/env/pacman/train.py \
    --algo a2c \
    --final-train
