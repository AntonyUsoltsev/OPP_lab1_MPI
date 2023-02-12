Compile command

    mpicc matrix_mpi_2.c -o matrix_mpi_2 -lm
Run command:

    mpirun -oversubscribe -np 4 ./matrix_mpi_2