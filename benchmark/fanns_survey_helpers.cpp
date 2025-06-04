#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <numeric>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <chrono>
#include "global_thread_counter.h"

std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading.\n";
        return {};
    }
    std::vector<std::vector<float>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;  // Read dimension
        std::vector<float> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float))) break;  // Read vector data
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading.\n";
        return {};
    }
    std::vector<std::vector<int>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;  // Read dimension
        std::vector<int> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int))) break;  // Read vector data
        dataset.push_back(move(vec));
    }
    file.close();
    return dataset;
}

std::vector<int> read_one_int_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<int> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::stringstream ss(line);
        int value;
        if (!(ss >> value)) {
            throw std::runtime_error("Non-integer or empty line at line " + std::to_string(line_number));
        }
        std::string extra;
        if (ss >> extra) {
            throw std::runtime_error("More than one value on line " + std::to_string(line_number));
        }
        result.push_back(value);
    }
    return result;
}

std::vector<std::vector<int>> read_multiple_ints_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::vector<int>> data;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::vector<int> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                if (!token.empty()) {
                    row.push_back(std::stoi(token));
                }
            } catch (...) {
                throw std::runtime_error("Invalid integer on line " + std::to_string(line_number));
            }
        }
        data.push_back(std::move(row));
    }
    return data;
}

std::vector<std::pair<int, int>> read_two_ints_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::pair<int, int>> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::stringstream ss(line);
        std::string first, second;
        if (!std::getline(ss, first, '-') || !std::getline(ss, second) || !ss.eof()) {
            throw std::runtime_error("Invalid format at line " + std::to_string(line_number));
        }
        try {
            int a = std::stoi(first);
            int b = std::stoi(second);
            result.emplace_back(a, b);
        } catch (...) {
            throw std::runtime_error("Invalid integer value at line " + std::to_string(line_number));
        }
    }
    return result;
}

void peak_memory_footprint() {
    unsigned iPid = (unsigned)getpid();

    std::cout << "PID: " << iPid << std::endl;

    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open())
    {
        std::cout << "memory information open error!" << std::endl;
    }
    std::string tmp;
    while (getline(info, tmp))
    {
        if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
            std::cout << tmp << std::endl;
    }
    info.close();
}


std::vector<int> parse_int_list(const std::string& input) {
    std::string cleaned = input;
    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(),
                  [](char c) { return c == '[' || c == ']'; }),
                  cleaned.end());

    std::vector<int> result;
    std::stringstream ss(cleaned);
    std::string token;

    while (std::getline(ss, token, ',')) {
        result.push_back(std::stoi(token));
    }

    return result;
}

void sort_by_attribute_and_remap(std::vector<std::vector<float>>& database_vectors, std::vector<int>& database_attributes, std::vector<std::vector<int>>& groundtruth) {
    // Step 1: Create indices
    std::vector<std::size_t> indices(database_attributes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Step 2: Sort indices by attribute values
    std::sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j) {
        return database_attributes[i] < database_attributes[j];
    });

    // Step 3: Build old-to-new index map
    std::unordered_map<int, int> old_to_new;
    for (std::size_t new_idx = 0; new_idx < indices.size(); ++new_idx) {
        old_to_new[indices[new_idx]] = static_cast<int>(new_idx);
    }

    // Step 4: Apply sorting to database_vectors and database_attributes
    std::vector<std::vector<float>> sorted_vectors(indices.size());
    std::vector<int> sorted_attributes(indices.size());

    for (std::size_t new_idx = 0; new_idx < indices.size(); ++new_idx) {
        std::size_t old_idx = indices[new_idx];
        sorted_vectors[new_idx] = std::move(database_vectors[old_idx]);
        sorted_attributes[new_idx] = database_attributes[old_idx];
    }

    // Step 5: Remap groundtruth indices
    for (auto& vec : groundtruth) {
        for (auto& idx : vec) {
            idx = old_to_new[idx];
        }
    }

    // Step 6: Replace original vectors
    database_vectors = std::move(sorted_vectors);
    database_attributes = std::move(sorted_attributes);
}


// Read current thread count from /proc/self/status
int get_thread_count() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("Threads:", 0) == 0) {
            return std::stoi(line.substr(8));
        }
    }
    return -1;
}

// Background monitor that updates peak thread count
void monitor_thread_count(std::atomic<bool>& done_flag) {
    while (!done_flag) {
        int current = get_thread_count();
        if (current > peak_threads) {
            peak_threads = current;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

