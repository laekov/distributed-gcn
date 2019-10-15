MPICXX ?= mpicxx
MPIHOME ?= $(shell dirname $(shell dirname $(shell which mpirun)))

ops.so : ops.o
	$(MPICXX) $< -shared -o $@


main : main.o ops.o
	nvcc -L$(MPIHOME)/lib main.o ops.o -o $@ -lmpi

main.o : main.cu 
	nvcc -I$(MPIHOME)/include $< -c -o $@ 

ops.o : ops.cu
	nvcc -I$(MPIHOME)/include $< -c -o $@ --compiler-options -fPIC

clean : 
	rm -f *.o *.so main
