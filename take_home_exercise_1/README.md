# Take Home Exercise - 1

## Overview

All the test cases have been hardcoded in [modified_pth_mat_vect_rand_split.c](modified_pth_mat_vect_rand_split.c).

For each of the 3 questions, there are 12 test cases.

Compile & run using:

```bash
make run
```

## a) GFlops/sec Calculation

GFlops (Giga Floating Point Operations per Second) calculations have been added for:

- Each thread calculation
- Overall execution for each case

### Formula

In a single row, each column has 2 flops. Therefore, the formula to calculate GFlops is:

```
GFlops = 2 * (num_of_rows) * (num_of_columns) / 10^9
```

## b) False Sharing Avoidance

To avoid the false sharing problem, padding was added to ensure each data point occupies a full cache line.

### Implementation Details

This was achieved by:

- Calculating the number of doubles required to fill a cache line
- Adjusting data storage such that only one data point is present per cache line

> **Note:** If not running on Linux, please adjust the cache line size variable according to your system.

### Cache Size Calculation

```c
cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
skip_size = cache_line_size / sizeof(double);
```

## c) Barrier Implementation

A barrier was added using the code from `pthread_barrier.c` from the labsheet.

- It is conditionally called after every row, based on the `use_barrier` flag.
