#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {

  std::ifstream file("./Electronics_5.json");
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file" << std::endl;
    return 1;
  }

  // Storing review count in unordered hashmap
  std::unordered_map<std::string, int> elaborate_review_count;

  std::string line;

  // Reading and processing concurrently
  while (std::getline(file, line)) {
    try {
      json record = json::parse(line);

      if (record.find("reviewText") != record.end()) {
        // Checking length of review only for lines that have reviewText field
        std::string review = record["reviewText"];

        if (review.length() >= 50) {
          elaborate_review_count[record["reviewerID"]] += 1;
        }
      }
    } catch (json::parse_error &e) {
      std::cerr << "JSON parse error: " << e.what() << " for line: " << line
                << std::endl;
    }
  }
  file.close();

  int ans{0};
  // Filtering review count
  for (auto itr : elaborate_review_count) {
    if (itr.second >= 5)
      ++ans;
      // std::cout << itr.first << " : " << itr.second << std::endl;
  }
  std::cout << ans << std::endl;
}
