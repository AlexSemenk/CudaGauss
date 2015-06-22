#ifndef CUDA_GAUSS_TEST_MODULE_GUARDIAN
#define CUDA_GAUSS_TEST_MODULE_GUARDIAN

#include "matrix.h"

using namespace std;

void test1(void solve(System<float>& a), int n);
void test2(void solve(System<float>& a), int n);

void test(void solve(System<float>& a), int n) {
	try {
		test1(solve, n);
		printf("test 1 passed with n = %d\n\n", n);
	} catch (const char* exc) {
		printf("Exception: %s\n", exc);
	}
	//try {
	//	test1(solve, n);
	//	printf("test 1 passed with n = %d\n\n", n);
	//} catch (const char* exc) {
	//	printf("Exception: %s\n", exc);
	//}
}

class MatrixTest  {
public:
	void test(void solve(System<float>& a), int n) {
		System<float> a(n);
		create(a);
		solve(a);
		if (!check(a)) {
			show(a);
			throw "test1 failed...";
		}
	}

	virtual void create(System<float>& a) {}
	virtual bool check(System<float>& a) {return false;}

	void show(System<float>& a) {
		ofstream file("test1.txt", ios::beg | ios::out);
		file << a << endl;
		file.close();
		system("notepad.exe test1.txt");
		if (remove("test1.txt") != 0)
			printf("Error deleting file test1.txt");
	}

};

#endif //CUDA_GAUSS_TEST_MODULE_GUARDIAN