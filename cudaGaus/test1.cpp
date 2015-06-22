#include "test.h"
#include "matrix.h"
#include <fstream>

/*			  

	|1 1 1 1 1 1|1|   |1 1 1 1 1 1|1|         |1 1 1 1 1 1|1|
	|1 2 2 2 2 2|2|   |0 1 1 1 1 1|1|         |0 1 1 1 1 1|1|
	|1 2 3 3 3 3|3|   |0 1 2 2 2 2|2|         |0 0 1 1 1 1|1|
	|1 2 3 4 4 4|4| ~ |0 1 2 3 3 3|3| ~ ... ~ |0 0 0 1 1 1|1|
	|1 2 3 4 5 5|5|   |0 1 2 3 4 4|4|         |0 0 0 0 1 1|1|
	|1 2 3 4 5 6|6|   |0 1 2 3 4 5|5|         |0 0 0 0 0 1|1|

*/

class Test1 : public MatrixTest {

	void create(System<float>& a) {
		size n = a.dim();
		for(index i=0; i<a.height; i++) {
			for (index j=0; j<a.width; j++) {
				a[i][j] = (float)min(i+1, j+1);
			}
		}
	}

	bool check(System<float>& a) {
		const float THRESHOLD = 0.0001f;
		size n = a.dim(), i, j;
		for(i=0; i<a.height; i++) {
			for(j=0; j<i; j++) {
				if (fabsf(a[i][j]-0.0f) > THRESHOLD) {
					printf("a[%d][%d] = %f != 0\n", i, j, a[i][j]);
					goto err;
				}
			} 
			for(j=i; j<a.width; j++) {
				if (fabsf(a[i][j]-1.0f) > THRESHOLD) {
					printf("a[%d][%d] = %f != 1\n", i, j, a[i][j]);
					goto err;
				}
			}
		}
		return true;
err:
		printf("Error in element a[%d][%d]\n", i, j);
		return false;
	}

};

void test1(void solve(System<float>& a), int n) {
	Test1().test(solve, n);
}