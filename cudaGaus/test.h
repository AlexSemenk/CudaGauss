#ifndef CUDA_GAUSS_TEST_MODULE_GUARDIAN
#define CUDA_GAUSS_TEST_MODULE_GUARDIAN

#include "matrix.h"

void test1(void func(System<float>& a), int n);
void test2(void func(System<float>& a), int n);

#endif//CUDA_GAUSS_TEST_MODULE_GUARDIAN