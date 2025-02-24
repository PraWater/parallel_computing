all:
	gcc -g -Wall -o modified_jacobi modified_jacobi.c -lm -fopenmp

modified_jacobi:
	gcc -g -Wall -o modified_jacobi modified_jacobi.c -lm -fopenmp

run: modified_jacobi
	@echo "Static Scheduling"
	@OMP_SCHEDULE=static ./modified_jacobi
	@echo ""
	@echo "------------------------------------------------------"
	@echo ""

	@echo "Dynamic Scheduling"
	@OMP_SCHEDULE=dynamic ./modified_jacobi
	@echo ""
	@echo "------------------------------------------------------"
	@echo ""

	@echo "Guided Scheduling"
	@OMP_SCHEDULE=guided ./modified_jacobi

clean:
	rm modified_jacobi
