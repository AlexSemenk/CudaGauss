#include "matrix.h"
#include <fstream>

using namespace std;

/*			  

	|1 1 1 1 1 1|1|   |1 1 1 1 1 1|1|         |1 1 1 1 1 1|1|
	|1 2 2 2 2 2|2|   |0 1 1 1 1 1|1|         |0 1 1 1 1 1|1|
	|1 2 3 3 3 3|3|   |0 1 2 2 2 2|2|         |0 0 1 1 1 1|1|
	|1 2 3 4 4 4|4| ~ |0 1 2 3 3 3|3| ~ ... ~ |0 0 0 1 1 1|1|
	|1 2 3 4 5 5|5|   |0 1 2 3 4 4|4|         |0 0 0 0 1 1|1|
	|1 2 3 4 5 6|6|   |0 1 2 3 4 5|5|         |0 0 0 0 0 1|1|

*/
void test1(void func(System<float>& a), int n) {
	const float THRESHOLD = 0.0001;
	System<float> a(n);
	// fill ...
	for(index i=0; i<a.height; i++) {
		for (index j=0; j<a.width; j++) {
			a[i][j] = (float)min(i+1, j+1);
		}
	}
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
