#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

int main() {
  std::ifstream file("./Electronics_5.json");
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file" << std::endl;
    return 1;
  }

  // Read all lines into memory first before processing
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();

  // Store review counts per reviewer in unordered hashmap
  std::unordered_map<std::string, int> elaborate_review_count;

  // Intermediate results to prevent race conditions
  std::vector<std::pair<std::string, int>> partial_results(lines.size());

// Parsing the JSON data on the GPU using OpenMP
#pragma omp target teams distribute parallel for map(to : lines)               \
    map(from : partial_results)
  for (size_t i = 0; i < lines.size(); i++) {
    try {
      json record = json::parse(lines[i]);

      // Skip is no matching words in lexicon
      if (record.find("reviewText") != record.end()) {
        std::string review = record["reviewText"];
        if (review.length() >= 50) {
          // Store in thread-local result
          partial_results[i] = std::make_pair(record["reviewerID"], 1);
        }
      }
    } catch (...) {
      // Error handling on GPU is limited - just store empty result
      partial_results[i] = std::make_pair("", 0);
    }
  }

  // Consolidate results on host
  for (const auto &result : partial_results) {
    if (!result.first.empty()) {
      elaborate_review_count[result.first] += result.second;
    }
  }

  // Count reviewers with at least 5 elaborate reviews
  int ans = 0;
  for (const auto &itr : elaborate_review_count) {
    if (itr.second >= 5)
      ++ans;
  }

  std::cout << ans << std::endl;
  return 0;
}
