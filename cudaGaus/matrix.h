#ifndef PARALLEL_GAUSS_PROJECT__MATRIX_MODULE__GUARDIAN
#define PARALLEL_GAUSS_PROJECT__MATRIX_MODULE__GUARDIAN

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <windows.h>

using std::cout;
using std::endl;
using std::ostream;

typedef size_t index;
typedef size_t size;

template<typename decimal> class Matrix {
public:
	decimal* array;
	size width;
	size height;
	Matrix(size w, size h) {
		width = w;
		height = h;
		array = new decimal[width*height];
	}
	Matrix(decimal* a, size w, size h) {
		array = a;
		width = w;
		height = h;
	}
	Matrix(Matrix& m) {
		width = m.width;
		height = m.height;
		array = new decimal[width*height];
		for(int i=0; i<m.height; i++) {
			for(int j=0; j<m.width; j++) {
				(*this)[i][j] = m[i][j];
			}
		}
	}
 	decimal* operator[](index i) {
		return (decimal*)((size)array + sizeof(decimal)*i*width);
	}
	template <typename T>
	friend ostream& operator<<(ostream& output, Matrix<T>& m) {
		for(int i=0; i<m.height; i++) {
			for(int j=0; j<m.width-1; j++) {
				output << m[i][j] << ' ';
			}
			output << m[i][m.width-1] << '\n';
		}
		return output;
	}
	~Matrix() {
		delete[] array;
	}
};

template<typename decimal> class System: public Matrix<decimal> {
public:
	System(System& s) : Matrix(s) {}
	System(size n) : Matrix(n+1, n) {}
	System(decimal* a, size n) : Matrix(a, n+1, n) {}
	size dim() {
		return height;
	}
};

template <typename decimal> decimal randRange(decimal inf, decimal sup) {
	decimal normalized = (decimal)rand()/(decimal)RAND_MAX;
	return (sup-inf)*normalized + inf;
}

template<typename decimal> void randMatrix(Matrix<decimal>& a, decimal min=0, decimal max=1) {
	for (index i=0, n=a.height; i<n; i++) {
		for (index j=0, m=a.width; j<m; j++) {
			a[i][j] = randRange(min, max);
		}
	}
}

template <typename T> void printfMatrix(const char* format, Matrix<T>& matrix) {	
	for (index i=0, n=matrix.height; i<n; i++) {
		for (index j=0, m=matrix.width; j<m; j++) {
			printf(format, matrix[i][j]);
		}
		cout << endl;
	}
	cout << endl;
}

template<typename T> void printToFile(const char* fileName, Matrix<T>& m) {
	std::ofstream file(fileName, std::ofstream::out | std::ios::trunc);
	file << m;
	file.close();
}

template<typename T> T abs(T arg) {
	return arg > 0 ? arg : -arg;
}

template<typename T> bool compare(Matrix<T>& m1, Matrix<T>& m2, T prec) {
	if (m1.width != m2.width || m1.height != m2.height){
		return false;
	}
	else {
		for(index i=0, n=m1.height; i<n; i++) {
			for(index j=0, m=m1.width; j<m; j++) {
				if (abs(m1[i][j]-m2[i][j]) > prec) {
					return false;
				}
			}
		}
		return true;
	}
}

#endif //PARALLEL_GAUSS_PROJECT__MATRIX_MODULE__GUARDIAN