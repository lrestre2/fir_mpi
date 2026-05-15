#!/bin/bash
#SBATCH --job-name=fir_serial
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/serial_%j.out
#SBATCH --error=logs/serial_%j.err

module load openmpi/gcc/64/4.1.8

mkdir -p logs

for N in 32 64 128 256 512; do
    for L in 100000 1000000 10000000; do
        echo -n "serial N=$N L=$L : "
        srun -n 1 ./fir_serial $N $L
    done
done