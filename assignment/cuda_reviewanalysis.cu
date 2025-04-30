#include <algorithm>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

static const int BLOCK_SIZE = 512;

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void scoresAccumulate(float *d_scores, int *d_indices,
                                 float *d_final_scores, int N) {
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N) {
    atomicAdd(&d_final_scores[d_indices[myID]], d_scores[myID]);
  }
}

std::vector<std::string> splitString(const std::string &str,
                                     const std::string &delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = str.find(delimiter);
  while (end != std::string::npos) {
    std::string token = str.substr(start, end - start);
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    tokens.push_back(token);
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }
  tokens.push_back(str.substr(start, str.length() - start));
  return tokens;
}

int main() {
  std::ifstream lexicon_file("./vader_lexicon.txt");
  if (!lexicon_file.is_open()) {
    std::cerr << "Error: Could not open file" << std::endl;
    return 1;
  }

  std::unordered_map<std::string, float> lexicon_score;

  std::string line;
  // Making hashmap for getting score of lexicon
  while (std::getline(lexicon_file, line)) {
    std::string key, value;
    std::stringstream s(line);
    std::getline(s, key, '\t');
    std::getline(s, value, '\t');
    try {
      lexicon_score[key] = std::stof(value);
    } catch (std::invalid_argument &e) {
      std::cerr << "Failed to convert to float for " << key
                << " with value = " << value << std::endl;
    }
  }
  lexicon_file.close();

  std::cout << "Processed " << lexicon_score.size() << " lexicons\n";

  std::ifstream file("./Electronics_5.json");
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file" << std::endl;
    return 1;
  }

  std::vector<float> h_scores;
  std::vector<int> h_indices;

  std::string delimiter = " ";
  int no_of_reviews{0};
  // Reading the file line by line while appending the scores of each word in the review and mapping the ASINs to integers.
  for (int i = 0; std::getline(file, line); ++i) {
    try {
      json record = json::parse(line);

      if (record.find("reviewText") != record.end()) {
        std::string review = record["reviewText"];
        std::vector<std::string> words = splitString(review, delimiter);
        bool check{false};
        for (const auto &word : words) {
          if (lexicon_score.find(word) != lexicon_score.end()) {
            check = true;
            h_scores.push_back(lexicon_score[word]);
            h_indices.push_back(no_of_reviews);
          }
        }
        if (check)
          ++no_of_reviews;
      }
    } catch (json::parse_error &e) {
      std::cerr << "JSON parse error: " << e.what() << " for line: " << line
                << std::endl;
    }
  }
  file.close();

  float *d_scores;
  int *d_indices;
  float *d_final_scores, *h_final_scores;

  // Device memory
  CUDA_CHECK_RETURN(cudaMalloc(&d_scores, h_scores.size() * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc(&d_indices, h_indices.size() * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMalloc(&d_final_scores, no_of_reviews * sizeof(float)));

  CUDA_CHECK_RETURN(cudaMemcpy(d_scores, h_scores.data(),
                               h_scores.size() * sizeof(float),
                               cudaMemcpyHostToDevice))
  CUDA_CHECK_RETURN(cudaMemcpy(d_indices, h_indices.data(),
                               h_indices.size() * sizeof(int),
                               cudaMemcpyHostToDevice))
  CUDA_CHECK_RETURN(
      cudaMemset(d_final_scores, 0, no_of_reviews * sizeof(float)))

  cudaEvent_t start, stop;
  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));

  CUDA_CHECK_RETURN(cudaEventRecord(start));

  int grid = ceil(h_scores.size() * 1.0 / BLOCK_SIZE);
  // Adds the scores together to find the review score.
  scoresAccumulate<<<grid, BLOCK_SIZE>>>(d_scores, d_indices, d_final_scores,
                                         h_scores.size());

  CUDA_CHECK_RETURN(cudaEventRecord(stop));
  CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Timetaken for scoresAccumulate: %f\n", milliseconds);
  CUDA_CHECK_RETURN(cudaEventDestroy(start));
  CUDA_CHECK_RETURN(cudaEventDestroy(stop));
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());

  // Host memory
  h_final_scores = (float *)malloc(no_of_reviews * sizeof(float));
  CUDA_CHECK_RETURN(cudaMemcpy(h_final_scores, d_final_scores,
                               no_of_reviews * sizeof(float),
                               cudaMemcpyDeviceToHost));

  // Counting reviews
  int positive{0}, negative{0}, neutral{0};
  for (int i = 0; i < no_of_reviews; ++i) {
    if (h_final_scores[i] > 0)
      ++positive;
    else if (h_final_scores[i] < 0)
      ++negative;
    else
      ++neutral;
  }

  std::cout << "Total no. of reviews: " << no_of_reviews << std::endl
            << "Positive: " << positive << " Negative: " << negative
            << " Neutral: " << neutral << std::endl;
}
