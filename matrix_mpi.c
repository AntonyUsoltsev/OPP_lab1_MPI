#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include "mpi.h"
#include <memory.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define N 6
#define t 0.01
#define eps 0.0001
#define RANK_ROOT 0

void print_matrix(const double *A, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%lf ", A[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(const double *vect, const int length) {
    for (int i = 0; i < length; i++) {
        printf("%lf ", vect[i]);
    }
    printf("\n");
}

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

//void transposition(const float *A, float *A_t) {
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            A_t[i * N + j] = A[j * N + i];
//        }
//    }
//}


void mult_matr_on_vect(const double *A, const int *arr_offset, int rank, const int width, const double *vect,
                       const int vect_len, double *res) {
    if (width != vect_len) {
        return;
    }
    for (int i = arr_offset[rank]; i < arr_offset[rank + 1]; i++) {
        double summ = 0;
        for (int j = 0; j < width; j++) {
            summ += A[i * width + j] * vect[j];
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
    summ = sqrt(summ);
    return summ;

}

int check(double vect_norm, double b_norm) {
    if (vect_norm / b_norm < eps) {
        return 0;
    }
    return 1;
}

int *set_offset(int size, int matrix_height) {
    int *offset_arr = calloc(size + 1, sizeof(int));
    if (size >= matrix_height) {
        for (int i = 0; i < size + 1; i++) {
            offset_arr[i] = i;
        }
        return offset_arr;
    }
    int a1 = matrix_height / size;
    int a2 = matrix_height % size;
    offset_arr[0] = 0;
    for (int i = 1; i < size - a2 + 1; i++) {
        offset_arr[i] = a1;
    }
    for (int i = size - a2 + 1; i < size + 1; i++) {
        offset_arr[i] = a1 + 1;
    }
    for (int i = 1; i < size + 1; i++) {
        offset_arr[i] = offset_arr[i] + offset_arr[i - 1];
    }

    return offset_arr;

}

int run(int comm_size, int comm_rank) {
    double *b = calloc(N, sizeof(double));
    fill_vector(b, N, (double) (N + 1));
    int *offset_arr = set_offset(comm_size, N);
    if (comm_rank == RANK_ROOT) {
        double *A = calloc(N * N, sizeof(double));
        fill_matrix(A, N, N);
        double b_norm = norm(b, N);
        // print_matrix(A, N, N);
        //   print_vect(b);

        double *x_prev = calloc(N, sizeof(double));
        fill_vector(x_prev, N, 0);

        double *x_next = calloc(N, sizeof(double));
        double *tmp = calloc(N, sizeof(double));
        // print_vect(x_prev);

        int flag = 1;



        for (int i = 0; i < comm_size + 1; i++) {
            printf("%d", offset_arr[i]);
        }
        puts("\n");

        for (int other_rank = 0; other_rank < comm_size; other_rank += 1) {
            if (other_rank == comm_rank) {
                continue;
            }

            const size_t other_offset = offset_arr[other_rank];
            const size_t other_chunk_size = offset_arr[other_rank + 1] - offset_arr[other_rank];

            MPI_Send((A + other_offset), (int) other_chunk_size, MPI_DOUBLE, other_rank, 0, MPI_COMM_WORLD);
        }
        const size_t offset = offset_arr[comm_rank];
        const size_t chunk_size = offset_arr[comm_rank + 1] - offset_arr[comm_rank];
        double *matr_chunk = (double *) calloc(chunk_size, sizeof(double ));
        memcpy(matr_chunk, A + offset, chunk_size * sizeof(double));
    } else {
        const size_t chunk_size = offset_arr[comm_rank + 1] - offset_arr[comm_rank];
        double *matr_chunk = (double *) calloc(chunk_size, sizeof(double ));
        MPI_Recv(matr_chunk, (int) chunk_size, MPI_DOUBLE, RANK_ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    const double partial_sum = vector_sum(vector_chunk, chunk_size);

}

int main(int argc, char **argv) {

    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == RANK_ROOT) {
        printf("Comm size: %d\n", size);
    }
    clock_t start = clock();
    run(size, rank);


    while (flag) {
        mult_matr_on_vect(A, offset_arr, rank, N, x_prev, N, x_next);
        //  print_vect(x_next);
        // sleep(5);
        diff_vector(x_next, N, b, N, x_next);
        //  print_vect(x_next);

        make_copy(x_next, N, tmp, N);

        mult_vect_on_num(x_next, N, t, x_next);
        // print_vect(x_next);

        diff_vector(x_prev, N, x_next, N, x_next);

        //print_vect(x_next);

        double tmp_norm = norm(tmp, N);
        flag = check(tmp_norm, b_norm);
        make_copy(x_next, N, x_prev, N);
    }
    clock_t end = clock();

    MPI_Finalize();

    printf("%ld\n", end - start);
    print_vector(x_next, N);

}