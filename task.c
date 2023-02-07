//#include <stdio.h>
//#include <stdlib.h>
//#include <stdbool.h>
//#include <unistd.h>
//#include "mpi.h"
//#include <memory.h>
//
//
//int main(int argc, char **argv) {
//    int size, rank;
//
//    MPI_Init(&argc, &argv);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    char name[MPI_MAX_PROCESSOR_NAME];
//    int resultlen;
//    printf("%d", MPI_Get_processor_name(name, &resultlen));
//
//    printf("Process %d\n", rank);
//    MPI_Finalize();
//    return 0;
//}





#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    sleep(world_rank);
    printf("Hello from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}