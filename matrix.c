#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>

#define N 5
#define t 0.01
#define eps 0.0001

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

void mult_matr_on_vect(const double *A, const int height, const int width, const double *vect, const int vect_len,
                       double *res) {
    if (width != vect_len) {
        return;
    }
    for (int i = 0; i < height; i++) {
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

int main(int argc, char **argv) {


    double *A = calloc(N * N, sizeof(double));
    double *b = calloc(N, sizeof(double));

    fill_matrix(A, N, N);

    fill_vector(b, N, (double) (N + 1));
    double b_norm = norm(b, N);
    print_matrix(A, N, N);
    //   print_vect(b);

    double *x_prev = calloc(N, sizeof(double));
    fill_vector(x_prev, N, 0);

    double *x_next = calloc(N, sizeof(double));
    double *tmp = calloc(N, sizeof(double));
    // print_vect(x_prev);

    int flag = 1;
    int size, rank;
    clock_t start = clock();
    while (flag) {
        mult_matr_on_vect(A, N, N, x_prev, N, x_next);
        //  print_vect(x_next);

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

    printf("%ld\n", end - start);
    print_vector(x_next, N);

}