#ifndef SRC_LENET5_CLASSIFY_LIB_H_
#define SRC_LENET5_CLASSIFY_LIB_H_

#include "common.h"

int argmax(float* arr, int size=10) {
	int max_arg = 0;
	float max = 0;
	for (int i = 0; i < size; i++) {
		if (arr[i] > max) {
			max_arg = i;
			max = arr[i];
		}
	}
	return max_arg;
}
double equal(int a, int b) {
	return (a == b) ? 1.0 : 0.0;
}



#endif
