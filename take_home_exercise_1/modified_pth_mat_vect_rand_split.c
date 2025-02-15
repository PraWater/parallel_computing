/* File:
 *     modified_pth_mat_vect_rand_split.c
 *
 * Purpose:
 *     Computes a parallel matrix-vector product.  Matrix
 *     is distributed by block rows.  Vectors are distributed by
 *     blocks.  This version uses a random number generator to
 *     generate A and x.  It also makes some small changes to
 *     the multiplication.  These are intended to improve
 *     performance and explicitly use temporary variables.
 *
 * Input:
 *     none
 *
 * Output:
 *     y: the product vector
 *     Elapsed time for the computation
 *
 * Compile:
 *     gcc -g -Wall -o modified_pth_mat_vect_rand_split modified_pth_mat_vect_rand_split.c -lpthread
 * Usage:
 *     ./modified_pth_mat_vect_rand_split
 *
 * Notes:
 *     1.  Local storage for A, x, y is dynamically allocated.
 *     2.  Number of threads (thread_count) should evenly divide
 *         m.  The program doesn't check for this.
 *     3.  We use a 1-dimensional array for A and compute subscripts
 *         using the formula A[i][j] = A[i*n + j]
 *     4.  Distribution of A, x, and y is logical:  all three are
 *         globally shared.
 *     5.  Compile with -DDEBUG for information on generated data
 *         and product.
 */

#include "timer.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Barrier functions */
typedef struct mylib_barrier_t {
  pthread_mutex_t count_lock;
  pthread_cond_t ok_to_proceed;
  int count;
} mylib_barrier_t;

void mylib_init_barrier(mylib_barrier_t *b);
void mylib_barrier(mylib_barrier_t *b, int num_threads);

/* Global variables */
int thread_count;
int m, n;
double *A;
double *x;
double *y;
long cache_line_size;
int skip_size; // Number of doubles that make up cache_line_size
bool use_padding;
bool use_barrier;
mylib_barrier_t barrier;

/* Serial functions */
void Gen_matrix(double A[], int m, int n);
void Gen_vector(double x[], int n);
void Print_matrix(char *title, double A[], int m, int n);
void Print_vector(char *title, double y[], double m);

/* Parallel function */
void *Pth_mat_vect(void *rank);

/* Structure for parameters of each test */
struct testcase {
  int m, n, thread_count;
  bool use_padding, use_barrier;
};

/*------------------------------------------------------------------*/
int main() {

  // If not using linux, hardcode the cache_line_size here
  cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);

  skip_size = cache_line_size / sizeof(double);
  long thread;
  pthread_t *thread_handles;
  mylib_init_barrier(&barrier);

  struct testcase testcases[36];

  // No padding and no barrier
  testcases[0].thread_count = 1; testcases[0].m = 8000000; testcases[0].n = 8; testcases[0].use_padding = false; testcases[0].use_barrier = false;
  testcases[1].thread_count = 1; testcases[1].m = 8000; testcases[1].n = 8000; testcases[1].use_padding = false; testcases[1].use_barrier = false;
  testcases[2].thread_count = 1; testcases[2].m = 8; testcases[2].n = 8000000; testcases[2].use_padding = false; testcases[2].use_barrier = false;
  testcases[3].thread_count = 2; testcases[3].m = 8000000; testcases[3].n = 8; testcases[3].use_padding = false; testcases[3].use_barrier = false;
  testcases[4].thread_count = 2; testcases[4].m = 8000; testcases[4].n = 8000; testcases[4].use_padding = false; testcases[4].use_barrier = false;
  testcases[5].thread_count = 2; testcases[5].m = 8; testcases[5].n = 8000000; testcases[5].use_padding = false; testcases[5].use_barrier = false;
  testcases[6].thread_count = 4; testcases[6].m = 8000000; testcases[6].n = 8; testcases[6].use_padding = false; testcases[6].use_barrier = false;
  testcases[7].thread_count = 4; testcases[7].m = 8000; testcases[7].n = 8000; testcases[7].use_padding = false; testcases[7].use_barrier = false;
  testcases[8].thread_count = 4; testcases[8].m = 8; testcases[8].n = 8000000; testcases[8].use_padding = false; testcases[8].use_barrier = false;
  testcases[9].thread_count = 8; testcases[9].m = 8000000; testcases[9].n = 8; testcases[9].use_padding = false; testcases[9].use_barrier = false;
  testcases[10].thread_count = 8; testcases[10].m = 8000; testcases[10].n = 8000; testcases[10].use_padding = false; testcases[10].use_barrier = false;
  testcases[11].thread_count = 8; testcases[11].m = 8; testcases[11].n = 8000000; testcases[11].use_padding = false; testcases[11].use_barrier = false;

  // Using padding and no barrier
  testcases[12].thread_count = 1; testcases[12].m = 8000000; testcases[12].n = 8; testcases[12].use_padding = true; testcases[12].use_barrier = false;
  testcases[13].thread_count = 1; testcases[13].m = 8000; testcases[13].n = 8000; testcases[13].use_padding = true; testcases[13].use_barrier = false;
  testcases[14].thread_count = 1; testcases[14].m = 8; testcases[14].n = 8000000; testcases[14].use_padding = true; testcases[14].use_barrier = false;
  testcases[15].thread_count = 2; testcases[15].m = 8000000; testcases[15].n = 8; testcases[15].use_padding = true; testcases[15].use_barrier = false;
  testcases[16].thread_count = 2; testcases[16].m = 8000; testcases[16].n = 8000; testcases[16].use_padding = true; testcases[16].use_barrier = false;
  testcases[17].thread_count = 2; testcases[17].m = 8; testcases[17].n = 8000000; testcases[17].use_padding = true; testcases[17].use_barrier = false;
  testcases[18].thread_count = 4; testcases[18].m = 8000000; testcases[18].n = 8; testcases[18].use_padding = true; testcases[18].use_barrier = false;
  testcases[19].thread_count = 4; testcases[19].m = 8000; testcases[19].n = 8000; testcases[19].use_padding = true; testcases[19].use_barrier = false;
  testcases[20].thread_count = 4; testcases[20].m = 8; testcases[20].n = 8000000; testcases[20].use_padding = true; testcases[20].use_barrier = false;
  testcases[21].thread_count = 8; testcases[21].m = 8000000; testcases[21].n = 8; testcases[21].use_padding = true; testcases[21].use_barrier = false;
  testcases[22].thread_count = 8; testcases[22].m = 8000; testcases[22].n = 8000; testcases[22].use_padding = true; testcases[22].use_barrier = false;
  testcases[23].thread_count = 8; testcases[23].m = 8; testcases[23].n = 8000000; testcases[23].use_padding = true; testcases[23].use_barrier = false;

  // No padding and using barrier
  testcases[24].thread_count = 1; testcases[24].m = 8000000; testcases[24].n = 8; testcases[24].use_padding = false; testcases[24].use_barrier = true;
  testcases[25].thread_count = 1; testcases[25].m = 8000; testcases[25].n = 8000; testcases[25].use_padding = false; testcases[25].use_barrier = true;
  testcases[26].thread_count = 1; testcases[26].m = 8; testcases[26].n = 8000000; testcases[26].use_padding = false; testcases[26].use_barrier = true;
  testcases[27].thread_count = 2; testcases[27].m = 8000000; testcases[27].n = 8; testcases[27].use_padding = false; testcases[27].use_barrier = true;
  testcases[28].thread_count = 2; testcases[28].m = 8000; testcases[28].n = 8000; testcases[28].use_padding = false; testcases[28].use_barrier = true;
  testcases[29].thread_count = 2; testcases[29].m = 8; testcases[29].n = 8000000; testcases[29].use_padding = false; testcases[29].use_barrier = true;
  testcases[30].thread_count = 4; testcases[30].m = 8000000; testcases[30].n = 8; testcases[30].use_padding = false; testcases[30].use_barrier = true;
  testcases[31].thread_count = 4; testcases[31].m = 8000; testcases[31].n = 8000; testcases[31].use_padding = false; testcases[31].use_barrier = true;
  testcases[32].thread_count = 4; testcases[32].m = 8; testcases[32].n = 8000000; testcases[32].use_padding = false; testcases[32].use_barrier = true;
  testcases[33].thread_count = 8; testcases[33].m = 8000000; testcases[33].n = 8; testcases[33].use_padding = false; testcases[33].use_barrier = true;
  testcases[34].thread_count = 8; testcases[34].m = 8000; testcases[34].n = 8000; testcases[34].use_padding = false; testcases[34].use_barrier = true;
  testcases[35].thread_count = 8; testcases[35].m = 8; testcases[35].n = 8000000; testcases[35].use_padding = false; testcases[35].use_barrier = true;

  for (int t = 0; t < 36; ++t) {
    /* Global variables being populated from testcase */
    thread_count = testcases[t].thread_count;
    m = testcases[t].m;
    n = testcases[t].n;
    use_padding = testcases[t].use_padding;
    use_barrier = testcases[t].use_barrier;

    printf("Testcase %d: thread_count =  %d, m = %d, n = %d, padding = "
           "%s, barrier = %s \n",
           t + 1, thread_count, m, n, use_padding ? "true" : "false",
           use_barrier ? "true" : "false");

#ifdef DEBUG
    printf("thread_count =  %d, m = %d, n = %d\n", thread_count, m, n);
#endif

    thread_handles = malloc(thread_count * sizeof(pthread_t));
    A = malloc(m * n * sizeof(double));
    x = malloc(n * sizeof(double));
    if (use_padding)
      y = malloc(m * sizeof(double) * skip_size);
    else
      y = malloc(m * sizeof(double));

    Gen_matrix(A, m, n);
#ifdef DEBUG
    Print_matrix("We generated", A, m, n);
#endif

    Gen_vector(x, n);
#ifdef DEBUG
    Print_vector("We generated", x, n);
#endif

    double start, end;

    GET_TIME(start);

    for (thread = 0; thread < thread_count; thread++)
      pthread_create(&thread_handles[thread], NULL, Pth_mat_vect,
                     (void *)thread);

    for (thread = 0; thread < thread_count; thread++)
      pthread_join(thread_handles[thread], NULL);

    GET_TIME(end);

    double gflops = 2e-9 * m * n / (end - start);
    printf("\nGFLOPS/sec: %f\n", gflops);
    printf("\n----------------------------------------------------------------------------------------------\n\n");

#ifdef DEBUG
    Print_vector("The product is", y, m);
#endif

    free(A);
    free(x);
    free(y);
    free(thread_handles);
  }

  return 0;
} /* main */

/*------------------------------------------------------------------
 * Function: Gen_matrix
 * Purpose:  Use the random number generator random to generate
 *    the entries in A
 * In args:  m, n
 * Out arg:  A
 */
void Gen_matrix(double A[], int m, int n) {
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i * n + j] = random() / ((double)RAND_MAX);
} /* Gen_matrix */

/*------------------------------------------------------------------
 * Function: Gen_vector
 * Purpose:  Use the random number generator random to generate
 *    the entries in x
 * In arg:   n
 * Out arg:  A
 */
void Gen_vector(double x[], int n) {
  int i;
  for (i = 0; i < n; i++)
    x[i] = random() / ((double)RAND_MAX);
} /* Gen_vector */

/*------------------------------------------------------------------
 * Function:       Pth_mat_vect
 * Purpose:        Multiply an mxn matrix by an nx1 column vector
 * In arg:         rank
 * Global in vars: A, x, m, n, thread_count
 * Global out var: y
 */
void *Pth_mat_vect(void *rank) {
  long my_rank = (long)rank;
  int i;
  int j;
  int local_m = m / thread_count;
  int my_first_row = my_rank * local_m;
  int my_last_row = my_first_row + local_m;
  register int sub = my_first_row * n;
  double start, finish;
  double temp;

#ifdef DEBUG
  printf("Thread %ld > local_m = %d, sub = %d\n", my_rank, local_m, sub);
#endif

  GET_TIME(start);
  for (i = my_first_row; i < my_last_row; i++) {
    int index = use_padding ? i * skip_size: i;
    y[index] = 0.0;
    for (j = 0; j < n; j++) {
      temp = A[sub++];
      temp *= x[j];     // 1st FLOP
      y[index] += temp; // 2nd FLOP
    }
    if (use_barrier)
      mylib_barrier(&barrier, thread_count);
  }
  GET_TIME(finish);
  double my_gflops = (2 * n * local_m / (finish - start)) * 1e-9;
  printf("Thread %ld > Elapsed time = %e seconds at %f GFLOPS/sec\n", my_rank,
         finish - start, my_gflops);

  return NULL;
} /* Pth_mat_vect */

/*------------------------------------------------------------------
 * Function:    Print_matrix
 * Purpose:     Print the matrix
 * In args:     title, A, m, n
 */
void Print_matrix(char *title, double A[], int m, int n) {
  int i, j;

  printf("%s\n", title);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++)
      printf("%6.3f ", A[i * n + j]);
    printf("\n");
  }
} /* Print_matrix */

/*------------------------------------------------------------------
 * Function:    Print_vector
 * Purpose:     Print a vector
 * In args:     title, y, m
 */
void Print_vector(char *title, double y[], double m) {
  int i;

  printf("%s\n", title);
  for (i = 0; i < m; i++)
    printf("%6.3f ", y[i]);
  printf("\n");
} /* Print_vector */

void mylib_init_barrier(mylib_barrier_t *b) {
  b->count = 0;
  pthread_mutex_init(&(b->count_lock), NULL);
  pthread_cond_init(&(b->ok_to_proceed), NULL);
}

void mylib_barrier(mylib_barrier_t *b, int num_threads) {
  pthread_mutex_lock(&(b->count_lock));
  b->count++;
  if (b->count == num_threads) {
    b->count = 0;
    pthread_cond_broadcast(&(b->ok_to_proceed));
  } else
    while (pthread_cond_wait(&(b->ok_to_proceed), &(b->count_lock)) != 0)
      ;
  pthread_mutex_unlock(&(b->count_lock));
}
