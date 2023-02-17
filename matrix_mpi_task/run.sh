#!/bin/bash

#PBS -l walltime=00:02:00
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=10000m,place=scatter
#PBS -m n
#PBS -o out.txt
#PBS -e err.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"

echo "Number of MPI process: $MPI_NP"
echo 'File $PBS_NODEFILE:'
cat $PBS_NODEFILE

echo "Using mpirun at `which mpirun`"
echo "Running $MPI_NP MPI processes"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./matrix_mpi