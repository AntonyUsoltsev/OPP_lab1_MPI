#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "mpi.h"
#include <math.h>
#include <time.h>
#include <memory.h>

#define N 1000
#define t 0.0001
#define eps 0.01
#define RANK_ROOT 0
#define CONTINUE 1
#define EXIT 0

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


void
mult_matr_on_vect(const double *matr_chunck, const int width, const int height, const double *vect, const int vect_len,
                  double *res) {
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
    // usleep(1000 * 10);
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
        return EXIT;
    }
    return CONTINUE;
}

int *set_offset(int size, int matrix_height) {
    int *offset_arr = calloc(size + 1, sizeof(int));
    if (size >= matrix_height) {
        for (int i = 0; i < size + 1; i++) {
            offset_arr[i] = i;
        }
        return offset_arr;
    }
    int quot = matrix_height / size;
    int remain = matrix_height % size;

    for (int i = 0; i < size + 1; i++) {
        if (i == 0) {
            offset_arr[i] = 0;
        } else if (i < size - remain + 1) {
            offset_arr[i] = offset_arr[i - 1] + quot;
        } else {
            offset_arr[i] = offset_arr[i - 1] + quot + 1;
        }
    }
    return offset_arr;
}


int run(int comm_size, int comm_rank) {
    double *b = calloc(N, sizeof(double));
    fill_vector(b, N, (double) (N + 1));
    double b_norm = norm(b, N);

    int *offset_arr = set_offset(comm_size, N);
    int comm_offset = (offset_arr[comm_rank + 1] - offset_arr[comm_rank]);

    double *A = NULL;
    double *x_prev = calloc(N, sizeof(double));
    fill_vector(x_prev, N, 0);

    double *x_next = calloc(N, sizeof(double));
    double *matr_chunk = NULL;

    if (comm_rank == RANK_ROOT) {
        A = calloc(N * N, sizeof(double));
        fill_matrix(A, N, N);
//        print_matrix(A, N, N);
//        print_vector(b, N);
//        print_vector(x_prev, N);
//        for (int i = 0; i < comm_size + 1; i++) {
//            printf("%d ", offset_arr[i]);
//        }
//        puts("\n");
        for (int other_rank = 0; other_rank < comm_size; other_rank++) {
            if (other_rank == comm_rank) {
                continue;
            }

            const size_t other_offset = offset_arr[other_rank];
            const size_t other_chunk_size = N * (offset_arr[other_rank + 1] - offset_arr[other_rank]);

            MPI_Send((A + other_offset * N), (int) other_chunk_size, MPI_DOUBLE, other_rank, 0, MPI_COMM_WORLD);
        }
        const size_t chunk_size = N * (offset_arr[comm_rank + 1] - offset_arr[comm_rank]);
        matr_chunk = (double *) calloc(chunk_size, sizeof(double));
        memcpy(matr_chunk, A, chunk_size * sizeof(double));

    } else {
        const size_t chunk_size = N * comm_offset;
        matr_chunk = (double *) calloc(chunk_size, sizeof(double));
        MPI_Recv(matr_chunk, (int) chunk_size, MPI_DOUBLE, RANK_ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int flag = 1;
    for (int i = 0; i < 1000 && flag == 1; i++) {
        double *res = (double *) calloc(comm_offset, sizeof(double));

        mult_matr_on_vect(matr_chunk, N, comm_offset, x_prev, N, res);

        if (comm_rank != RANK_ROOT) {
            MPI_Send(res, comm_offset, MPI_DOUBLE, RANK_ROOT, 0, MPI_COMM_WORLD);

        } else {
            memcpy(x_next, res, comm_offset * sizeof(double));

            for (int other_rank = 1; other_rank < comm_size; other_rank++) {
                size_t chunk_size = offset_arr[other_rank + 1] - offset_arr[other_rank];
                double *res_2 = calloc(chunk_size, sizeof(double));
                MPI_Recv(res_2, chunk_size, MPI_DOUBLE, other_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                memcpy(x_next + offset_arr[other_rank], res_2,chunk_size * sizeof(double));
                free(res_2);
            }
            //  print_vector(x_next, N);
        }
        if (comm_rank == RANK_ROOT) {
            diff_vector(x_next, N, b, N, x_next);

            double *tmp = calloc(N, sizeof(double));
            make_copy(x_next, N, tmp, N);
            double tmp_norm = norm(tmp, N);
            if (!check(tmp_norm, b_norm)) {
                for (int other_rank = 1; other_rank < comm_size; other_rank++) {
                    MPI_Send(x_next, N, MPI_DOUBLE, other_rank, EXIT, MPI_COMM_WORLD);
                }
                break;
            }
            free(tmp);

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
    print_vector(x_prev, N);
    free(A);
    free(b);
    free(x_next);
    free(x_prev);
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

    const double start_time = MPI_Wtime();
    run(size, rank);
    const double end_time = MPI_Wtime();

    MPI_Finalize();

    printf("Process %d took %.2lfs\n", rank, end_time - start_time);
}
