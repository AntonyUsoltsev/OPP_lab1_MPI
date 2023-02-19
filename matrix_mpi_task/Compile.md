Compile command

    mpicc matrix_mpi.c -o matrix_mpi -lm -Wpedantic -Werror -Wall -O3 --std=c99

Run command:
    
    mpirun -oversubscribe -np 4 ./matrix_mpi
