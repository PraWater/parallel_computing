#include <cuda.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static const int BLOCK_SIZE = 256;
static const int HASHMAP_SIZE = 1024;

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }

// Device string functions
__device__ char *d_strstr(const char *haystack, const char *needle) {
  auto match_prefix = [](const char *s, const char *prefix) {
    while ((*prefix != '\0') && (*s == *prefix)) {
      s++, prefix++;
    }
    return (*prefix == '\0');
  };
  do {
    if (match_prefix(haystack, needle)) {
      return const_cast<char *>(haystack);
    }
  } while (*(haystack++) != '\0');
  return (*needle == '\0') ? const_cast<char *>(haystack) : nullptr;
}

__device__ char *d_strncpy(char *dst, const char *src, size_t n) {
  size_t i = 0;
  auto ret = dst;
  for (; i < n && *src != '\0'; i++, src++, dst++) {
    *dst = *src;
  }
  for (; i < n; i++, dst++) {
    *dst = '\0';
  }
  return ret;
}

__device__ int hash(const int n) { return n % HASHMAP_SIZE; }

__global__ void parseJson(char **strings, char *asins, char *ratings,
                          int num_strings) {
  int myID = threadIdx.x + blockIdx.x * blockDim.x;
  if (myID < num_strings) {
    char *rating_pointer = d_strstr(strings[myID], "\"overall\":");
    if (rating_pointer != nullptr) {
      rating_pointer += 11; // Skip "overall":
      ratings[myID] = rating_pointer[0];
    } else {
      ratings[myID] = '?'; // Mark as not found
    }

    char *asin_pointer = d_strstr(strings[myID], "\"asin\":");
    if (asin_pointer != nullptr) {
      asin_pointer += 9; // Skip "asin": "
      d_strncpy(&asins[myID * 11], asin_pointer, 10);
      asins[myID * 11 + 10] = '\0'; // Ensure null termination
    } else {
      d_strncpy(&asins[myID * 11], "NOTFOUND", 8);
      asins[myID * 11 + 8] = '\0';
    }
  }
}

// Added privatization to prevent race conditions
__global__ void ratingsAdd(int *d_ratings, int *d_indices,
                           float *d_processed_ratings, int *d_ratings_count,
                           int N) {
  __shared__ int s_indices[HASHMAP_SIZE];
  __shared__ float s_processed_ratings[HASHMAP_SIZE];
  __shared__ int s_ratings_count[HASHMAP_SIZE];

  int local_id = threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Default values for the shared variables
  for (int i = local_id; i < HASHMAP_SIZE; i += blockDim.x) {
    s_indices[i] = -1;
    s_processed_ratings[i] = 0.0f;
    s_ratings_count[i] = 0;
  }
  __syncthreads();

  // Use atomicAdd to fill shared variables
  for (int idx = global_id; idx < N; idx += total_threads) {
    int id = d_indices[idx];
    float rating = d_ratings[idx];
    int h = hash(id);

    while (true) {
      int old = atomicCAS(&s_indices[h], -1, id);
      if (old == -1 || old == id) {
        atomicAdd(&s_processed_ratings[h], rating);
        atomicAdd(&s_ratings_count[h], 1);
        break;
      }
      h = (h + 1) % HASHMAP_SIZE;
    }
  }
  __syncthreads();

  // Shared variables -> Global variables
  for (int i = local_id; i < HASHMAP_SIZE; i += blockDim.x) {
    int id = s_indices[i];
    if (id != -1) {
      atomicAdd(&d_processed_ratings[id], s_processed_ratings[i]);
      atomicAdd(&d_ratings_count[id], s_ratings_count[i]);
    }
  }
}

__global__ void ratingsDivide(float *d_processed_ratings, int *d_ratings_count,
                              int N) {
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N) {
    d_processed_ratings[myID] /= (float)d_ratings_count[myID];
  }
}

int main() {
  std::string filename = "./Electronics_5.json";

  std::vector<std::string> lines;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file" << std::endl;
    return 1;
  }

  // Read the file line by line using C++ strings
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();

  int num_ratings = lines.size();

  char *d_string_data;
  std::vector<char *> d_lines(num_ratings);
  char **d_line_ptrs;

  size_t total_string_size = 0;
  for (const auto &str : lines) {
    total_string_size += str.length() + 1; // +1 for null terminator
  }

  std::cout << "Allocating " << total_string_size << " bytes for string data"
            << std::endl;

  // Allocate memory on device for strings
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_string_data, total_string_size));

  size_t offset = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    size_t len = lines[i].length() + 1;
    CUDA_CHECK_RETURN(cudaMemcpy(d_string_data + offset, lines[i].c_str(), len,
                                 cudaMemcpyHostToDevice));
    d_lines[i] = d_string_data + offset;
    offset += len;
  }

  CUDA_CHECK_RETURN(
      cudaMalloc((void **)&d_line_ptrs, num_ratings * sizeof(char *)));

  CUDA_CHECK_RETURN(cudaMemcpy(d_line_ptrs, d_lines.data(),
                               num_ratings * sizeof(char *),
                               cudaMemcpyHostToDevice));

  char *d_asins, *d_ratings_chars;
  CUDA_CHECK_RETURN(
      cudaMalloc((void **)&d_asins,
                 num_ratings * 11)); // ASIN (10 chars + null terminator)
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_ratings_chars,
                               num_ratings)); // Rating (single char per item)

  CUDA_CHECK_RETURN(cudaMemset(d_asins, 0, num_ratings * 11));
  CUDA_CHECK_RETURN(cudaMemset(d_ratings_chars, 0, num_ratings));

  std::vector<char> h_asins(num_ratings * 11, 0);
  std::vector<char> h_ratings_chars(num_ratings, 0);

  cudaEvent_t start, stop;
  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));

  CUDA_CHECK_RETURN(cudaEventRecord(start));

  // GPU to parse JSON and collect all the ratings along with ASINs
  int grid = (num_ratings + BLOCK_SIZE - 1) / BLOCK_SIZE;
  parseJson<<<grid, BLOCK_SIZE>>>(d_line_ptrs, d_asins, d_ratings_chars,
                                  num_ratings);

  CUDA_CHECK_RETURN(cudaEventRecord(stop));
  CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Timetaken for parsing JSON in GPU: %f\n", milliseconds);
  CUDA_CHECK_RETURN(cudaEventDestroy(start));
  CUDA_CHECK_RETURN(cudaEventDestroy(stop));

  // Copy results back to host
  CUDA_CHECK_RETURN(cudaMemcpy(h_asins.data(), d_asins, num_ratings * 11,
                               cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaMemcpy(h_ratings_chars.data(), d_ratings_chars,
                               num_ratings, cudaMemcpyDeviceToHost));

  std::unordered_map<std::string, int> asin_to_id;
  std::unordered_map<int, std::string> id_to_asin;
  std::vector<int> ratings;
  std::vector<int> indices;
  ratings.reserve(num_ratings);
  indices.reserve(num_ratings);

  // Mapping the ASINs to integers
  for (int i = 0; i < num_ratings; i++) {
    std::string asin(&h_asins[i * 11]);
    if (asin != "NOTFOUND") {
      int rating = h_ratings_chars[i] - '0';
      if (rating > 5 || rating < 0)
        rating = 0;
      ratings.push_back(rating);

      if (asin_to_id.find(asin) != asin_to_id.end()) {
        int id = asin_to_id[asin];
        indices.push_back(id);
      } else {
        int id = asin_to_id.size();
        asin_to_id[asin] = id;
        id_to_asin[id] = asin;
        indices.push_back(id);
      }
    }
  }

  int no_of_products = asin_to_id.size();
  std::cout << "Processed " << no_of_products << " products." << std::endl;

  CUDA_CHECK_RETURN(cudaFree(d_asins));
  CUDA_CHECK_RETURN(cudaFree(d_ratings_chars));
  CUDA_CHECK_RETURN(cudaFree(d_string_data));
  CUDA_CHECK_RETURN(cudaFree(d_line_ptrs));

  // Allocate device memory
  int *d_ratings;
  int *d_indices;
  float *d_processed_ratings;
  int *d_ratings_count;
  CUDA_CHECK_RETURN(cudaMalloc(&d_ratings, ratings.size() * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMalloc(&d_indices, indices.size() * sizeof(int)));
  CUDA_CHECK_RETURN(
      cudaMalloc(&d_processed_ratings, no_of_products * sizeof(float)));
  CUDA_CHECK_RETURN(
      cudaMemset(d_processed_ratings, 0, no_of_products * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc(&d_ratings_count, no_of_products * sizeof(int)));
  CUDA_CHECK_RETURN(
      cudaMemset(d_ratings_count, 0, no_of_products * sizeof(int)));

  // Copy data to device
  CUDA_CHECK_RETURN(cudaMemcpy(d_ratings, ratings.data(),
                               ratings.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_indices, indices.data(),
                               indices.size() * sizeof(int),
                               cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));

  CUDA_CHECK_RETURN(cudaEventRecord(start));

  grid = (ratings.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  ratingsAdd<<<grid, BLOCK_SIZE>>>(d_ratings, d_indices, d_processed_ratings,
                                   d_ratings_count, ratings.size());

  CUDA_CHECK_RETURN(cudaEventRecord(stop));
  CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
  milliseconds = 0;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Timetaken for ratingsAdd: %f\n", milliseconds);
  CUDA_CHECK_RETURN(cudaEventDestroy(start));
  CUDA_CHECK_RETURN(cudaEventDestroy(stop));

  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));

  CUDA_CHECK_RETURN(cudaEventRecord(start));

  grid = (no_of_products + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
  std::vector<float> h_processed_ratings(no_of_products);
  CUDA_CHECK_RETURN(cudaMemcpy(h_processed_ratings.data(), d_processed_ratings,
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
    printf("Index: %d, Asin: %s, Value: %f\n", maxi, id_to_asin[maxi].c_str(),
           h_processed_ratings[maxi]);
    h_processed_ratings[maxi] = 0;
  }

  // Free device memory
  CUDA_CHECK_RETURN(cudaFree(d_ratings));
  CUDA_CHECK_RETURN(cudaFree(d_processed_ratings));
  CUDA_CHECK_RETURN(cudaFree(d_indices));
  CUDA_CHECK_RETURN(cudaFree(d_ratings_count));

  return 0;
}
