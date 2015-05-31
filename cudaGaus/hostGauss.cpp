#ifndef PARALLEL_GAUSS_PROECT__HOST_GAUSS_MODULE__GUARDIAN
#define PARALLEL_GAUSS_PROECT__HOST_GAUSS_MODULE__GUARDIAN

#include "matrix.h"
#include <iostream>

template <typename decimal> void hostForwardGauss(System<decimal> &a) {
	size n = a.dim();
	for(index k=0; k<n-1; k++) {
		for(index i=k+1; i<n; i++) {
			decimal mul = a[i][k] / a[k][k];
			for(index j=k; j<n+1; j++) {
				a[i][j] = a[i][j] - mul * a[k][j];
			}
		}
	}
}

#endif