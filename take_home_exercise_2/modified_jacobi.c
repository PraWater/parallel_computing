#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

  /*float a[10][10], b[10], x[10], xn[10], sum, e;*/
  /*int i, j, n, flag = 0, key;*/
  /*printf("\nThis program illustrates Gauss-Jacobi method to solve system of
   * "*/
  /*       "AX=B\n");*/
  /*printf("\nEnter the dimensions of coefficient matrix n: ");*/
  /*scanf("%d", &n);*/
  /*printf("\nEnter the elements of matrix A:\n");*/
  /*for (i = 0; i < n; i++) {*/
  /*  for (j = 0; j < n; j++) {*/
  /*    scanf("%f", &a[i][j]);*/
  /*  }*/
  /*}*/
  /*printf("\nEnter the elements of matrix B:\n");*/
  /*for (i = 0; i < n; i++)*/
  /*  scanf("%f", &b[i]);*/
  /*printf("\nThe system of linear equations:\n");*/
  /*for (i = 0; i < n; i++) {*/
  /*  printf("\n(%.2f)x1+(%.2f)x2+(%.2f)x3=(%.2f)\n", a[i][0], a[i][1],
   * a[i][2],*/
  /*         b[i]);*/
  /*}*/

  // Defining the size of the matrix
  int n = 720;
  float e = 0.001;
  int key, flag = 0;

  // Dynamically allocate memory for matrix A
  float **a = (float **)malloc(n * sizeof(float *));
  for (int i = 0; i < n; i++) {
    a[i] = (float *)malloc(n * sizeof(float));
  }

  // Dynamically allocate memory for vectors B, X, and Xn
  float *b = (float *)malloc(n * sizeof(float));
  float *x = (float *)malloc(n * sizeof(float));
  float *xn = (float *)malloc(n * sizeof(float));

  // Hardcoded Coefficient Matrix A (Diagonally dominant for convergence)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        a[i][j] = 4.0; // Diagonal elements
      } else if (j == i - 1 || j == i + 1) {
        a[i][j] = -1.0; // Neighbors
      } else {
        a[i][j] = 0.0;
      }
    }
  }

  // Hardcoded B vector
  for (int i = 0; i < n; i++) {
    b[i] = 100.0;
  }

  // Initial approximation for X
  for (int i = 0; i < n; i++) {
    x[i] = 0.0;
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);

  /* No loop carried dependency for the loop below
   * Justification: none of the variables except sum is being written to. And
   * sum is initialised to 0 at the beginning of every loop, so it doesn't
   * depend on the previous loop. Flag is a global variable, but when one of the
   * loop flags it, we can just stop all the loops, so it doesn't depend
   *
   * Scheduling: Static is better for this loop. Since each iteration roughly
   * takes the same amount of time. */
#pragma omp parallel for reduction(+ : flag) schedule(runtime)
  for (int i = 0; i < n; i++) {
    float sum = 0;

    /* No loop carried dependency for the loop below
     * Justification: this is just a sum of multiple elements calculated in a
     * loop. It can be parallelized using the reduction clause in omp */
#pragma omp simd reduction(+ : sum)
    for (int j = 0; j < n; j++) {
      sum += fabs(a[i][j]);
    }

    sum -= fabs(a[i][i]);
    /*if (fabs(a[i][i] < sum)) {*/
    if (fabs(a[i][i]) < sum) {
      flag += 1;
    }
  }

  gettimeofday(&end, NULL);
  double time_in_ms = end.tv_usec - start.tv_usec;
  time_in_ms /= 1000;
  printf("Time taken for diagonally dominant check = %f ms\n", time_in_ms);

  if (flag >= 1) {
    printf("\nThe system of linear equations are not diagonally dominant\n");
    exit(1);
  } else {
    /*printf("\nThe system of linear equations are diagonally dominant\n");*/
    /*printf("\nEnter the initial approximations: ");*/
    /*for (i = 0; i < n; i++) {*/
    /*  // x[i]=0;*/
    /*  printf("\nx%d=", (i + 1));*/
    /*  scanf("%f", &x[i]);*/
    /*}*/
    /*printf("\nEnter the error tolerance level:\n");*/
    /*scanf("%f", &e);*/
  }
/*printf("x[1]\t\tx[2]\t\tx[3]");*/
/*printf("\n");*/
gettimeofday(&start, NULL);
#pragma omp parallel
  {
    key = 0;
    while (key < n - 1) {
      key = 0;

      /* No loop carried dependency for the loop below
       * Justification: sum is a local variable for each iteration and the only
       * other variable being written to is not being used by another loop. Once
       * again key value is being incremented in each loop, so we can use omp
       * reduction for that 
       *
       * Scheduling: Dynamic is better for this. All iterations don't have to add to
       * the key variable. So the iterations which do have to increment, will take
       * longer. This causes imbalance of time spent per iteration. */
#pragma omp for reduction(+ : key) schedule(runtime)
      for (int i = 0; i < n; i++) {
        float sum = b[i];

        /* No loop carried dependency for the loop below
         * Justification: This is also an example of reduction and can be done
         * using omp reduction. Checks for a condition and adds to the variable
         */
#pragma omp simd reduction(- : sum)
        for (int j = 0; j < n; j++)
          if (j != i)
            sum -= a[i][j] * x[j];

        xn[i] = sum / a[i][i];
        if (fabs(x[i] - xn[i]) < e) {
          ++key;
        }
      }

      if (key == n) {
        break;
      }
      /*printf("%f\t %f\t %f\t", xn[0], xn[1], xn[2]);*/
      /* No loop carried dependency for the loop below
       * Justification: assignment in a for loop, can be parallel because in
       * each iteration nothing is being used from any of the other loops
       *
       * Scheduling: Static is better, because each iteration takes the same
       * amount of time */
#pragma omp for schedule(runtime)
      for (int i = 0; i < n; i++) {
        x[i] = xn[i];
      }
    }
  }
  gettimeofday(&end, NULL);
  time_in_ms = end.tv_usec - start.tv_usec;
  time_in_ms /= 1000;
  printf("Time taken for approximating answer = %f ms\n", time_in_ms);
  /*printf("\nAn approximate solution to the given system of equations is\n");*/
  /*for (i = 0; i < n; i++) {*/
  /*  printf("\nx[%d]=%f\n", (i + 1), x[i]);*/
  /*}*/
  return 0;
}
