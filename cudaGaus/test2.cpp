#include "matrix.h"
#include <fstream>

using namespace std;

/*			  

	|1 -1  1 -1  1 -1| 1|         |1 -1  1 -1  1 -1| 1|
	|1  0  0  0  0  0| 0|         |0  1 -1  1 -1  1|-1|
	|1  0  1 -1  1 -1| 1|         |0  0  1 -1  1 -1| 1|
	|1  0  1  0  0  0| 0| ~ ... ~ |0  0  0  1 -1  1|-1|
	|1  0  1  0  1 -1| 1|         |0  0  0  0  1 -1| 1|
	|1  0  1  0  1  0| 0|         |0  0  0  0  0  1|-1|

*/

namespace test2space {
	void create(System<float>& a);
	bool check(System<float>& a);
	void show(System<float>& a);
}

void test2(void solve(System<float>& a), int n) {
	using namespace test2space;	
	System<float> a(n);
	create(a);
	solve(a);
	if (!check(a)) {
		show(a);
		throw "test2 failed...";
	}
}

namespace test2space {

	void create(System<float>& a) {
		size n = a.dim();
		for (index d=0; d<n; d++) {
			if (d % 2 == 0) {
				a[d][d] = 1;
				for (index i=d+1; i<n; i++)	
					a[i][d] = 1;
				for (index j=d+1; j<n+1; j++)	
					a[d][j] = ((j-d)%2 == 0) ? 1.0f : -1.0f;
			} else {
				a[d][d] = 0;
				for (index i=d+1; i<n; i++)	
					a[i][d] = 0.0f;
				for (index j=d+1; j<n+1; j++)	
					a[d][j] = 0.0f;
			}
		}
	}

	bool check(System<float>& a) {
		const float THRESHOLD = 0.0001f;
		size n = a.dim(), i, j;
		for(i=0; i<n; i++) {
			for(j=0; j<i; j++) {
				if (fabsf(a[i][j]-0.0f) > THRESHOLD) {
					return false;
				}
			}
			for(j=i; j<n+1; j++) {
				if ((j-i)%2 == 0) {
					if (fabsf(a[i][j]-1.0f) > THRESHOLD) goto err;
					else continue;
				} else {
					if (fabsf(a[i][j]+1.0f) > THRESHOLD) goto err;
					else continue;
				}
			}
		}
		return true;
err:
		printf("Error in element a[%d][%d]\n", i, j);
		return false;
	}

	void show(System<float>& a) {
		ofstream file("test2.txt", ios::beg | ios::out);
		file << a << endl;
		file.close();
		system("notepad.exe test2.txt");
		if (remove("test2.txt") != 0)
			printf( "Error deleting file test2.txt" );
	}

}