cmake_minimum_required(VERSION 3.16.3)
project(lab1 C)

set(CMAKE_C_STANDARD 99)
#include_directories(/usr/include/mpi/)
#set(CMAKE_C_COMPILER "/usr/bin/mpicc")
add_executable(lab1  matrix.c)
