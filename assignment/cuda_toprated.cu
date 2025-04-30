#include <cuda.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdlib.h>
#include <string>
#include <vector>

using json = nlohmann::json;

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void ratingsAdd(float *d_ratings, int *d_indexes,
                           float *d_processed_ratings, int *d_ratings_count,
                           int N) {
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N) {
    atomicAdd(&d_processed_ratings[d_indexes[myID]], d_ratings[myID]);
    atomicAdd(&d_ratings_count[d_indexes[myID]], 1);
  }
}

__global__ void ratingsDivide(float *d_processed_ratings, int *d_ratings_count,
                              int N) {
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N) {
    d_processed_ratings[myID] /= (float)d_ratings_count[myID];
  }
}

static const int BLOCK_SIZE = 256;

int main(int argc, char *argv[]) {
  std::ifstream file("./Electronics_5.json");
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file" << std::endl;
    return 1;
  }

  std::map<std::string, int> asin_to_id;
  std::map<int, std::string> id_to_asin;
  std::vector<float> ratings;
  std::vector<int> indexes;
  ratings.reserve(1e7);
  indexes.reserve(1e7);

  std::string line;
  // Reading line by line and mapping ASIN to an integer
  while (std::getline(file, line)) {
    try {
      json record = json::parse(line);

      std::string asin = record["asin"];
      float rating = record["overall"];
      ratings.push_back(rating);

      if (asin_to_id.find(asin) != asin_to_id.end()) {
        int id = asin_to_id[asin];
        indexes.push_back(id);
      } else {
        int id = asin_to_id.size();
        asin_to_id[asin] = id;
        id_to_asin[id] = asin;
        indexes.push_back(id);
      }
    } catch (json::parse_error &e) {
      std::cerr << "JSON parse error: " << e.what() << " for line: " << line
                << std::endl;
    }
  }
  file.close();

  int no_of_products = asin_to_id.size();
  std::cout << "Processed " << no_of_products << " products." << std::endl;

  // Allocate device memory
  float *d_ratings;
  int *d_indices;
  float *d_processed_ratings;
  int *d_ratings_count;
  float *h_processed_ratings;
  CUDA_CHECK_RETURN(cudaMalloc(&d_ratings, ratings.size() * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc(&d_indices, indexes.size() * sizeof(int)));
  CUDA_CHECK_RETURN(
      cudaMalloc(&d_processed_ratings, no_of_products * sizeof(float)));
  CUDA_CHECK_RETURN(
      cudaMemset(d_processed_ratings, 0, no_of_products * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc(&d_ratings_count, no_of_products * sizeof(int)));
  CUDA_CHECK_RETURN(
      cudaMemset(d_ratings_count, 0, no_of_products * sizeof(int)));

  // Copy data to device
  CUDA_CHECK_RETURN(cudaMemcpy(d_ratings, ratings.data(),
                               ratings.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_indices, indexes.data(),
                               indexes.size() * sizeof(int),
                               cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));

  CUDA_CHECK_RETURN(cudaEventRecord(start));

  int grid = ceil(ratings.size() * 1.0 / BLOCK_SIZE);
  // Accumulating the ratings
  ratingsAdd<<<grid, BLOCK_SIZE>>>(d_ratings, d_indices, d_processed_ratings,
                                   d_ratings_count, ratings.size());

  CUDA_CHECK_RETURN(cudaEventRecord(stop));
  CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Timetaken for ratingsAdd: %f\n", milliseconds);
  CUDA_CHECK_RETURN(cudaEventDestroy(start));
  CUDA_CHECK_RETURN(cudaEventDestroy(stop));


  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));

  CUDA_CHECK_RETURN(cudaEventRecord(start));

  grid = ceil(no_of_products * 1.0 / BLOCK_SIZE);
  // Averaging the ratings
  ratingsDivide<<<grid, BLOCK_SIZE>>>(d_processed_ratings, d_ratings_count,
                                      no_of_products);
  CUDA_CHECK_RETURN(cudaEventRecord(stop));
  CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
  milliseconds = 0;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Timetaken for ratingsDivide: %f\n", milliseconds);
  CUDA_CHECK_RETURN(cudaEventDestroy(start));
  CUDA_CHECK_RETURN(cudaEventDestroy(stop));


  // Copy results back to host
  h_processed_ratings = (float *)malloc(no_of_products * sizeof(float));
  CUDA_CHECK_RETURN(cudaMemcpy(h_processed_ratings, d_processed_ratings,
                               no_of_products * sizeof(float),
                               cudaMemcpyDeviceToHost));

  // Selecting the top 10
  for (int i = 0; i < 10; ++i) {
    int maxi = 0;
    for (int j = 1; j < no_of_products; ++j) {
      if (h_processed_ratings[j] > h_processed_ratings[maxi]) {
        maxi = j;
      }
    }
    printf("Index: %d, Asin: %s, Value: %f\n", maxi, id_to_asin[maxi].data(),
           h_processed_ratings[maxi]);
    h_processed_ratings[maxi] = 0;
  }

  free(h_processed_ratings);
  CUDA_CHECK_RETURN(cudaFree(d_ratings));
  CUDA_CHECK_RETURN(cudaFree(d_processed_ratings));
  CUDA_CHECK_RETURN(cudaFree(d_indices));

  return 0;
}
