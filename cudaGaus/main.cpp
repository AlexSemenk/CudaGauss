#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include "matrix.h"
#include "test.h"

#include "hostGauss.cpp"

using namespace std;

extern "C" void global_gauss_2d(System<float>& system);
extern "C" void global_gauss_3d(System<float>& system);
extern "C" void texture_gauss_2d(System<float>& system);

#define N 1000


void test1(void func(System<float>& a), int n);

int main(int argc, char* argv[]) {
	try {
		test1(global_gauss_3d, N);
		printf("test 1 passed with n = %d\n", N);
	} catch (const char* exc) {
		printf("Exception: %s\n", exc);
	}
}