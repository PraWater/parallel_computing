#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NO_OF_EVENTS 120
#define EVENT_ID 12

char **split_to_lines(char *buffer, int *line_count) {
  // Count the number of lines first
  int total_lines = 1;
  for (int i = 0; buffer[i] != '\0'; i++) {
    if (buffer[i] == '\n') {
      total_lines++;
    }
  }

  // Allocate memory for line pointers
  char **lines = malloc(total_lines * sizeof(char *));
  if (!lines) {
    *line_count = 0;
    return NULL;
  }

  // Copy and split the paragraph
  *line_count = 0;

  // Use strtok to split the paragraph
  char *line = strtok(buffer, "\n");
  while (line != NULL) {
    // Allocate memory for each line and copy it
    lines[*line_count] = strdup(line);
    (*line_count)++;

    // Get next line
    line = strtok(NULL, "\n");
  }

  return lines;
}

const char *getfield(char *line, int num) {
  const char *tok;
  for (tok = strtok(line, ";"); tok && *tok; tok = strtok(NULL, ";\n")) {
    if (!--num)
      return tok;
  }
  return NULL;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, comm_sz, line_count;
  int *event_count;
  char **lines;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, "./BGL_2k.log_structured.csv", MPI_MODE_RDONLY,
                MPI_INFO_NULL, &fp);

  MPI_Offset total_size;
  MPI_File_get_size(fp, &total_size);

  MPI_Offset chunk_size = total_size / comm_sz;
  MPI_Offset start = rank * chunk_size;
  MPI_Offset end = (rank + 1) * chunk_size;

  // Adjust start to the next newline (to avoid splitting lines)
  if (rank != 0) {
    char c;
    MPI_File_read_at(fp, start - 1, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    while (c != '\n' && start < total_size) {
      start++;
      MPI_File_read_at(fp, start - 1, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    }
  }

  // Adjust end to the next newline
  if (rank != comm_sz - 1) {
    --end;
    char c;
    MPI_File_read_at(fp, end, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    while (c != '\n' && end < total_size) {
      end++;
      MPI_File_read_at(fp, end, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    end++;
  }

  const MPI_Offset read_size = end - start;
  char *buffer = (char *)malloc(read_size + 1);
  MPI_File_read_at(fp, start, buffer, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
  buffer[read_size] = '\0';
  MPI_File_close(&fp);

  lines = split_to_lines(buffer, &line_count);
  free(buffer);

  event_count = calloc(NO_OF_EVENTS, sizeof(int));

#pragma omp parallel for
  for (int i = 0; i < line_count; ++i) {
    // Memory problem caused here when using openmp. Somehow gets fixed when
    // printing \n
    /*printf("\n");*/
    const char *event_code = getfield(lines[i], EVENT_ID);
    free(lines[i]);
    const int code = atoi(event_code + 1);
#pragma omp critical
    ++event_count[code - 1];
  }

  int *total_event_count = calloc(NO_OF_EVENTS, sizeof(int));

  MPI_Reduce(event_count, total_event_count, NO_OF_EVENTS, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    // Process 0 prints to event_count.txt
    MPI_Send(total_event_count, NO_OF_EVENTS, MPI_INT, 1, 0, MPI_COMM_WORLD);
    FILE *f = fopen("./event_count.txt", "w+");
    for (int i = 0; i < NO_OF_EVENTS; ++i) {
      fprintf(f, "E%d: %d\n", i + 1, total_event_count[i]);
    }
    fclose(f);
  } else if (rank == 1) {
    // Process 1 prints to top_10.txt
    MPI_Recv(total_event_count, NO_OF_EVENTS, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    FILE *f = fopen("./top_10.txt", "w+");
    for (int i = 0; i < 10; ++i) {
      int max = 0;
      for (int j = 1; j < NO_OF_EVENTS; ++j) {
        if (total_event_count[j] > total_event_count[max])
          max = j;
      }
      fprintf(f, "E%d: %d\n", max + 1, total_event_count[max]);
      total_event_count[max] = 0;
    }
    fclose(f);
  }

  MPI_Finalize();
  return 0;
}
