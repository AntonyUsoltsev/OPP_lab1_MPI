#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <stdbool.h>

#define N 16500
#define t 0.00001f
#define eps 0.00001f

#define RANK_ROOT 0
#define CONTINUE 1
#define EXIT 0

void print_vector_double(const double *vect, const int length) {
    for (int i = 0; i < length; i++) {
        printf("%lf ", vect[i]);
    }
    printf("\n");
}

//void print_vector_int(const int *vect, const int length) {
//    for (int i = 0; i < length; i++) {
//        printf("%d ", vect[i]);
//    }
//    printf("\n");
//}


void fill_vector(double *vector, const int length, const double fill_value) {
    for (size_t i = 0; i < length; i++) {
        vector[i] = fill_value;
    }
}


void fill_matrix(double *A, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == j)
                A[i * width + j] = 2.0f;
            else
                A[i * width + j] = 1.0f;
        }
    }
}


double *create_matrix() {
    unsigned long long matr_size = (unsigned long long) N * N;
    double *A = calloc(matr_size, sizeof(double));
    fill_matrix(A, N, N);
    return A;
}


void mult_matr_on_vect(const double *matr_chunck, const int width, const int height, const double *vect,
                       const int vect_len, double *res) {
    if (width != vect_len) {
        return;
    }
    for (int i = 0; i < height; i++) {

        double summ = 0;
        for (int j = 0; j < width; j++) {
            summ += matr_chunck[i * width + j] * vect[j];
        }
        res[i] = summ;
    }
}


void diff_vector(const double *vect_1, const int len_1, const double *vect_2, const int len_2, double *res) {
    if (len_1 != len_2) {
        return;
    }
    for (int i = 0; i < len_1; i++) {
        res[i] = vect_1[i] - vect_2[i];
    }
}


void mult_vect_on_num(const double *vect, const int vect_len, const double number, double *res) {
    for (int i = 0; i < vect_len; i++) {
        res[i] = vect[i] * number;
    }
}


void make_copy(const double *vect_1, const int len_1, double *vect_2, const int len_2) {
    if (len_1 != len_2) {
        return;
    }
    for (int i = 0; i < len_1; i++) {
        vect_2[i] = vect_1[i];
    }
}


double norm(const double *vect, const int vect_len) {
    double summ = 0;
    for (int i = 0; i < vect_len; i++) {
        summ += vect[i] * vect[i];
    }
    return summ;

}


int check_eps(double vect_norm, double b_norm) {
    if (vect_norm / b_norm < eps * eps) {
        return EXIT;
    }
    return CONTINUE;
}


int *set_chunk_sizes(int size, int matrix_height) {
    int *chunk_sizes = calloc(size, sizeof(int));
    int quot = matrix_height / size;
    int remain = matrix_height % size;

    for (int i = 0; i < size; i++) {
        if (i < size - remain) {
            chunk_sizes[i] = quot;
        } else {
            chunk_sizes[i] = quot + 1;
        }
    }
    return chunk_sizes;
}


int *set_offset(int size, const int *chunk_size_arr) {
    int *offset_arr = calloc(size, sizeof(int));
    offset_arr[0] = 0;
    for (int i = 1; i < size; i++) {
        offset_arr[i] = offset_arr[i - 1] + chunk_size_arr[i - 1];
    }
    return offset_arr;
}


int *set_matr_chunk_sizes(const int *chunk_size_arr, int size, int param) {
    int *matrix_chunk_size_arr = calloc(size, sizeof(int));
    for (int i = 0; i < size; i++) {
        matrix_chunk_size_arr[i] = chunk_size_arr[i] * param;
    }
    return matrix_chunk_size_arr;
}


int *set_matr_offset(const int *offset_arr, int size, int param) {
    int *matrix_offset_arr = calloc(size, sizeof(int));
    for (int i = 0; i < size; i++) {
        matrix_offset_arr[i] = offset_arr[i] * param;
    }
    return matrix_offset_arr;
}


double *send_matrix_from_root(int comm_rank, int comm_size, double *A, int *chunk_size_arr, int *offset_arr) {
    int *matrix_chunk_size_arr = set_matr_chunk_sizes(chunk_size_arr, comm_size, N);
    int *matrix_offset_arr = set_matr_offset(offset_arr, comm_size, N);

    double *matrix_chunk = calloc(matrix_chunk_size_arr[comm_rank], sizeof(double));

    MPI_Scatterv(A, matrix_chunk_size_arr, matrix_offset_arr, MPI_DOUBLE, matrix_chunk,
                 matrix_chunk_size_arr[comm_rank],
                 MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    return matrix_chunk;
}


int run(int comm_size, int comm_rank) {

    int *chunk_size_arr = set_chunk_sizes(comm_size, N);
    int *offset_arr = set_offset(comm_size, chunk_size_arr);

    int comm_chunk_size = chunk_size_arr[comm_rank];

    double *b = calloc(comm_chunk_size, sizeof(double));
    fill_vector(b, comm_chunk_size, (double) (N + 1));

    double local_b_norm = norm(b, comm_chunk_size);
    double global_b_norm = 0;
    MPI_Reduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, RANK_ROOT, MPI_COMM_WORLD);

    double *A = NULL;
    double *x_prev = calloc(N, sizeof(double));
    fill_vector(x_prev, N, 0);

    double *x_next = calloc(N, sizeof(double));

    double *matrix_chunk = NULL;

    if (comm_rank == RANK_ROOT) {
        A = create_matrix();
    }

    matrix_chunk = send_matrix_from_root(comm_rank, comm_size, A, chunk_size_arr, offset_arr);

    while (true) {
        double *res = (double *) calloc(comm_chunk_size, sizeof(double));

        mult_matr_on_vect(matrix_chunk, N, comm_chunk_size, x_prev, N, res);

        diff_vector(res, comm_chunk_size, b, comm_chunk_size, res);

        double local_Ax_b_norm = norm(res, comm_chunk_size);
        double global_Ax_b_norm = 0;

        MPI_Reduce(&local_Ax_b_norm, &global_Ax_b_norm, 1, MPI_DOUBLE, MPI_SUM, RANK_ROOT, MPI_COMM_WORLD);

        MPI_Gatherv(res, comm_chunk_size, MPI_DOUBLE, x_next, chunk_size_arr, offset_arr, MPI_DOUBLE,
                    RANK_ROOT, MPI_COMM_WORLD);

        if (comm_rank == RANK_ROOT) {

            if (!check_eps(global_Ax_b_norm, global_b_norm)) {
                for (int other_rank = 1; other_rank < comm_size; other_rank++) {
                    MPI_Send(x_prev, N, MPI_DOUBLE, other_rank, EXIT, MPI_COMM_WORLD);
                }
                break;
            }

            mult_vect_on_num(x_next, N, t, x_next);
            diff_vector(x_prev, N, x_next, N, x_next);
            make_copy(x_next, N, x_prev, N);

            for (int other_rank = 1; other_rank < comm_size; other_rank++) {
                MPI_Send(x_next, N, MPI_DOUBLE, other_rank, CONTINUE, MPI_COMM_WORLD);
            }

        } else {
            MPI_Status status;
            MPI_Recv(x_prev, N, MPI_DOUBLE, RANK_ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == EXIT)
                break;
        }


    }
    if (comm_rank == RANK_ROOT) {
        //print_vector_double(x_prev,N);
        printf("Process № %d x[0] value is: %lf\n", comm_rank, x_prev[0]);
    }
    free(A);
    free(b);
    free(x_next);
    free(x_prev);
    free(matrix_chunk);
    free(chunk_size_arr);
    free(offset_arr);
    return 1;
}


int main(int argc, char **argv) {

    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == RANK_ROOT) {

        printf("Comm size: %d\n", size);
    }
    if (size > N) {
        size = N;
    }
    const double start_time = MPI_Wtime();
    run(size, rank);
    const double end_time = MPI_Wtime();

    MPI_Finalize();

    printf("Process %d took %.4lfs\n", rank, end_time - start_time);
}