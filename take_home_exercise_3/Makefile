all:
	mpicc -g -Wall -std=c99 -lm -o histogram_mpi histogram_mpi.c

histogram_mpi:
	mpicc -g -Wall -std=c99 -lm -o histogram_mpi histogram_mpi.c

run: histogram_mpi
	mpiexec -n 8 ./histogram_mpi
	@cat hist_0.txt

clean:
	rm histogram_mpi
	rm hist_*.txt
