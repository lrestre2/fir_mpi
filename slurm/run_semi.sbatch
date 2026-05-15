#!/bin/bash
#SBATCH --job-name=fir_semi
#SBATCH --partition=defq
#SBATCH --nodes=8
#SBATCH --ntasks=192
#SBATCH --time=00:30:00
#SBATCH --output=logs/semi_%j.out
#SBATCH --error=logs/semi_%j.err

module load openmpi/gcc/64/4.1.8

mkdir -p logs

N=128
L=10000000

for P in 1 2 4 8 16 32 64 128 192; do
    echo -n "semi N=$N L=$L P=$P : "
    srun -n $P ./fir_semi_parallel $N $L
done