Compile command

    mpicc matrix_mpi.c -o matrix_mpi -lm -Wpedantic -Werror -Wall -O3

Run command:
    
    mpirun -oversubscribe -np 4 ./matrix_mpi