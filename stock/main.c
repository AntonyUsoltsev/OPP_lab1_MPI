#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include "mpi.h"
#include <memory.h>

#define VECTOR_SIZE 29

#define RANK_ROOT 0

void vector_fill(double *vector, size_t size, double fill_value) {
    for (size_t i = 0; i < size; i += 1) {
        vector[i] = fill_value;
    }
}


double vector_sum(const double *vector, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; i += 1) {
        usleep(1000 * 100);
        sum += vector[i];
    }
    return sum;
}


size_t vector_chunk_size(size_t size, int comm_size, int rank) {
    const size_t default_size = size / comm_size;
    const bool is_last = comm_size - 1 == rank;
    if (!is_last) {
        return default_size;
    }

    return default_size + size % comm_size;
}


size_t vector_chunk_offset(size_t size, int comm_size, int rank) {
    return (size / comm_size) * rank;
}


int run(int comm_size, int comm_rank) {
    double *vector = NULL;

    const size_t chunk_size = vector_chunk_size(VECTOR_SIZE, comm_size, comm_rank);
    double *vector_chunk = (double *) calloc(chunk_size, sizeof(*vector_chunk));

    printf("Rank: %d, Chunk size: %llu\n", comm_rank, (unsigned long long) chunk_size);

    if (RANK_ROOT == comm_rank) {
        vector = (double *) calloc(VECTOR_SIZE, sizeof(*vector_chunk));
        vector_fill(vector, VECTOR_SIZE, 1.0);

        for (int other_rank = 0; other_rank < comm_size; other_rank += 1) {
            if (other_rank == comm_rank) {
                continue;
            }

            const size_t other_offset = vector_chunk_offset(VECTOR_SIZE, comm_size, other_rank);
            const size_t other_chunk_size = vector_chunk_size(VECTOR_SIZE, comm_size, other_rank);

            MPI_Send((vector + other_offset), (int) other_chunk_size, MPI_DOUBLE, other_rank, 0, MPI_COMM_WORLD);
        }

        const size_t offset = vector_chunk_offset(VECTOR_SIZE, comm_size, comm_rank);
        memcpy(vector_chunk, vector + offset, chunk_size * sizeof(double));
    } else {
        MPI_Recv(vector_chunk, (int) chunk_size, MPI_DOUBLE, RANK_ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    const double partial_sum = vector_sum(vector_chunk, chunk_size);
    double sum = 0.0;

    MPI_Reduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, RANK_ROOT, MPI_COMM_WORLD);

    if (RANK_ROOT == comm_rank) {
        printf("Sum: %.2lf\n", sum);
    }

    free(vector);
    free(vector_chunk);

    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (RANK_ROOT == rank) {
        printf("Comm size: %d\n", size);
    }

    const double start_time_s = MPI_Wtime();
    const int exit_code = run(size, rank);
    const double end_time_s = MPI_Wtime();

    MPI_Finalize();

    printf("Process %d took %.2lfs\n", rank, end_time_s - start_time_s);
    return exit_code;
}