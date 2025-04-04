all:
	mpicc -g -Wall -fopenmp -o log_omp_mpi log_omp_mpi.c

log_omp_mpi:
	mpicc -g -Wall -fopenmp -o log_omp_mpi log_omp_mpi.c

run: log_omp_mpi
	mpiexec -n 4 ./log_omp_mpi

clean:
	rm log_omp_mpi event_count.txt top_10.txt
