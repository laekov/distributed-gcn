#include "mpi.h"
#include "cuda.h"
#include <cstdio>

int comm_rank, comm_size;
int succ, prev;

MPI_Request reqs[2];
MPI_Status status[2];

extern "C" { 
void init() {
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	succ = (comm_rank + 1) % comm_size;
	prev = (comm_rank - 1 + comm_size) % comm_size;
	printf("Hi, I am %d/%d, my succ is %d, my prev is %d\n", comm_rank, 
			comm_size, succ, prev);
}

int rank() {
	return comm_rank;
}

int size() {
	return comm_size;
}

void allred(void* pi, void* po, size_t sz) {
	MPI_Allreduce(pi, po, sz, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void ring_pass(void* vp_send, void* vp_recv, size_t sz, int iter_id) {
	MPI_Isend(vp_send, sz, MPI_FLOAT, succ, (iter_id << 16) | succ,
			MPI_COMM_WORLD, reqs + 0);
	MPI_Irecv(vp_recv, sz, MPI_FLOAT, prev, (iter_id << 16) | comm_rank, 
			MPI_COMM_WORLD, reqs + 1);
}

void wait() {
	MPI_Waitall(2, reqs, status);
}

void finalize() {
	MPI_Finalize();
}

};  // extern C
