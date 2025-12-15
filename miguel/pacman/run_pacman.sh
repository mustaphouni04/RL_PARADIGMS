#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /fhome/pmlai09/miguel/env/pacman # working directory
#SBATCH -t 7-00:00 # Runtime in D-HH:MM
#SBATCH -p tfg # Partition to submit to
#SBATCH --mem 8192 # 8GB memory.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 
sleep 5

python /fhome/pmlai09/miguel/env/pacman/train.py \
    --algo a2c \
    --final-train

nvidia-smi
