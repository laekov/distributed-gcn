#include "ops.h"
#include <cstdio>

int main() {
	init();
	int comm_rank = rank();
	float* a = new float[10];
	float* b;
	cudaMalloc(&b, 10 * sizeof(float));
	if (comm_rank == 0) {
		a[0] = 1;
		cudaMemcpy(b, a, 10 * sizeof(float), cudaMemcpyHostToDevice);
		MPI_Send(b, 10, MPI_FLOAT, 1, 111, MPI_COMM_WORLD);
	} else {
		MPI_Status s;
		MPI_Recv(b, 10, MPI_FLOAT, 0, 111, MPI_COMM_WORLD, &s);
		cudaMemcpy(a, b, 10 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("recv %.3lf\n", a[0]);
	}
	finalize();
}
