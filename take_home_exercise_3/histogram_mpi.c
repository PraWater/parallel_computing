/* Assumption: Number of rows is divisible by number of processes */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Change size of matrix here */
#define NO_OF_ROWS 16
#define NO_OF_COLUMNS 16

int main(void) {
  int comm_sz;          /* Number of processes */
  int my_rank;          /* My process rank */
  int histogram[10];    /* My copy of the histogram */
  int rows_per_process; /* No. of rows each process averages */
  int **my_rows;        /* The rows to be averaged by this process */
  int *my_avgs;         /* Averages calculated by this process */

  /* Start up MPI */
  MPI_Init(NULL, NULL);

  /* Get the number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  /* Get my rank among all the processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Initializing histogram to 0 */
  memset(histogram, 0, sizeof(histogram));

  rows_per_process = NO_OF_ROWS / comm_sz;

  /* Generating the rows of the matrix to be averaged randomly.
   * Here we can generate the whole matrix, but except for the rows it averages,
   * nothing else will be used by this particular process. So we are not
   * generating the rest to save time. If we have to generate the entire matrix,
   * we have to make the seed for rand is same across all processes */
  my_rows = malloc(rows_per_process * sizeof(int *));
  srand(time(NULL) * my_rank);
  for (int i = 0; i < rows_per_process; ++i) {
    my_rows[i] = malloc(NO_OF_COLUMNS * sizeof(int));
    for (int j = 0; j < NO_OF_COLUMNS; ++j)
      my_rows[i][j] = rand() % 9;
  }

  my_avgs = calloc(rows_per_process, sizeof(int));

  /* Calculating avgs for my rows */
  for (int i = 0; i < rows_per_process; ++i) {
    for (int j = 0; j < NO_OF_COLUMNS; ++j)
      my_avgs[i] += my_rows[i][j];
    my_avgs[i] = ceilf((float)my_avgs[i] / NO_OF_COLUMNS);
  }

  /* Sending my rows' avgs to all other processes */
  for (int process_no = 0; process_no < comm_sz; ++process_no) {
    if (process_no ==
        my_rank) // Instead of sending messages to self, adding to histogram
      for (int i = 0; i < rows_per_process; ++i)
        ++histogram[my_avgs[i]];
    else {
      for (int i = 0; i < rows_per_process; ++i)
        MPI_Send(&my_avgs[i], 1, MPI_INT, process_no, 0, MPI_COMM_WORLD);
    }
  }

  /* Receiving avgs and adding to histogram from all other processes */
  for (int i = 0; i < rows_per_process * (comm_sz - 1); ++i) {
    int avg;
    MPI_Recv(&avg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    ++histogram[avg];
  }

  /* Writing histogram to file */
  char filename[] = "hist_x.txt";
  filename[5] = my_rank + '0';
  FILE *histFile = fopen(filename, "w");

  fprintf(histFile, "Histogram calculated by process %d\n", my_rank);
  for (int i = 0; i < 10; ++i) {
    fprintf(histFile, "%d: ", i);
    for (int j = 0; j < histogram[i]; ++j)
      fprintf(histFile, "x");
    fprintf(histFile, "\n");
  }

  fclose(histFile);

  /* Shut down MPI */
  MPI_Finalize();

  return 0;
} /* main */
