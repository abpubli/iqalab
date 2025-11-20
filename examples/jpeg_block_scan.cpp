#include "iqalab/iqalab.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <iomanip>

namespace fs = std::filesystem;

struct FileScore
{
    std::string path;
    double      score = 0.0;
};

std::vector<fs::path> scan_file_or_directory(const fs::path& input)
{
    std::vector<fs::path> result;
    if (fs::is_regular_file(input)) {
        FileScore fsItem;
        result.push_back(input);
    } else if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (!entry.is_regular_file())
                continue;
            const fs::path& p = entry.path();
            if (!iqa::detect_image_type1(p))
                continue;
            result.push_back(p);
        }
    } else {
        throw std::runtime_error("Input is neither regular file nor directory: " +
                                 input.string());
    }
    sort(result.begin(), result.end());
    return result;
}

void print_to_console(const FileScore &fsItem)
{
    std::cout << fsItem.path << " : "
        << std::setprecision(6) << fsItem.score << "\n";
}

void write_to_csv(const std::vector<FileScore>& files,
                  const std::string& csvPath)
{
    std::ofstream out(csvPath, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open CSV file for writing: " + csvPath);
    }

    // two columns without header
    for (const auto& fsItem : files) {
        out << "\"" << fsItem.path << "\""
            << "," << std::setprecision(10) << fsItem.score << "\n";
    }
}

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <image_or_dir>\n"
                  << "  " << argv[0] << " <image_or_dir> <out.csv>\n";
        return 1;
    }

    fs::path inputPath = argv[1];
    const bool toCsv   = (argc == 3);
    std::string csvPath;
    if (toCsv) {
        csvPath = argv[2];
    }

    try {
        std::vector<fs::path> paths = scan_file_or_directory(inputPath);
        std::vector<FileScore> scores;
        for (int counter = 0; counter < paths.size(); ++counter) {
            const auto& path = paths[counter];
            std::cout << counter+1 << "/" << paths.size() << ": ";
            FileScore fsItem;
            fsItem.path  = path.string();
            fsItem.score = iqa::blocking_score_from_file(fsItem.path);
            print_to_console(fsItem);
            scores.push_back(fsItem);
        }
        if (toCsv)
            write_to_csv(scores, csvPath);

        if (paths.empty()) {
            std::cerr << "No image files found.\n";
            return 0;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}