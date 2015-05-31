#include "matrix.h"
#include <fstream>

using namespace std;

/*			  

	|1 -1  1 -1  1 -1| 1|
	|1  0  0  0  0  0| 0|
	|1  0  1 -1  1 -1| 1|
	|1  0  1  0  0  0| 0| ~ ... ~
	|1  0  1  0  1 -1| 1|
	|1  0  1  0  1  0| 0|

*/
void test2(void func(System<float>& a), int n) {
	const float THRESHOLD = 0.0001;
	System<float> a(n);
	// fill ...
	for (int d=0; d<n; d++) {
		if (d % 2 == 0) {
		} else {
			
		}
	}
	// func ...
	func(a);
	// test ...
	for(index i=0; i<a.height; i++) {
		for(index j=0; j<i; j++) {
			if (fabsf(a[i][j]-0.0f) > THRESHOLD) {
				goto exc;
			}
		}
		for(index j=i; j<a.width; j++) {
			if (fabsf(a[i][j]-1.0f) > THRESHOLD) {
				goto exc;
			}
		}
	}
	return;
exc:
	// debug ...
	ofstream file("out.txt", ios::beg | ios::out);
	file << a << endl;
	file.close();
	system("notepad.exe out.txt");
	// clean ...
	throw "test1 failed...";
}

