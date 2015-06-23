#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include "matrix.h"

#include "hostGauss.cpp"

using namespace std;

extern "C" void global_gauss_3d1(System<float>& system);
extern "C" void global_gauss_3d2(System<float>& system);
void test(void solve(System<float>& a), int n);

#define N 1000

int main(int argc, char* argv[]) {
	test(global_gauss_3d1, N);
	//test(global_gauss_3d2, N);
}