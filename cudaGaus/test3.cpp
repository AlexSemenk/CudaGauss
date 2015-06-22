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

namespace test3space {
	void create(System<float>& a);
	bool check(System<float>& a);
	void show(System<float>& a);
}

void test3(void solve(System<float>& a), int n) {
	using namespace test3space;	
	System<float> a(n);
	create(a);
	solve(a);
	if (!check(a)) {
		show(a);
		throw "test2 failed...";
	}
}

namespace test3space {

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
		size n = a.dim();
		for(index i=0; i<n; i++) {
			for(index j=0; j<i; j++) {
				if (fabsf(a[i][j]-0.0f) > THRESHOLD) {
					return false;
				}
			}
			for(index j=i; j<n+1; j++) {
				if ((j-i)%2 == 0) {
					if (fabsf(a[i][j]-1.0f) > THRESHOLD) return false;
					else continue;
				} else {
					if (fabsf(a[i][j]+1.0f) > THRESHOLD) return false;
					else continue;
				}
			}
		}
		return true;
	}

	void show(System<float>& a) {
		ofstream file("test3.txt", ios::beg | ios::out);
		file << a << endl;
		file.close();
		system("notepad.exe test3.txt");
		if (remove("test3.txt") != 0)
			printf( "Error deleting file test2.txt" );
	}

}