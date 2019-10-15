#include "mpi.h"

extern "C"  {
void init();
void finalize();
int rank();
void ring_pass(void* vp_send, void* vp_recv, size_t sz, int iter_id);

};
