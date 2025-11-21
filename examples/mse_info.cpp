#include "iqalab/mse.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "iqalab/utils/file_grouping.hpp"
#include "iqalab/utils/path_utils.hpp"

namespace fs = std::filesystem;
using namespace iqa;
using namespace iqa::utils;

struct CliOptions {
    fs::path ref;
    fs::path dist;
};

bool parse_args(int argc, char** argv, CliOptions& opts)
{
    if (argc < 4) {
        std::cerr << "Usage:\n";
        std::cerr << "  mse_info <ref_file> <dist_file>\n";
        std::cerr << "  mse_info <ref_dir>  <dist_dir>\n";
        return false;
    }

    opts.ref  = argv[1];
    opts.dist = argv[2];

    return true;
}

static bool copy_or_fail(const fs::path& src, const fs::path& dst)
{
    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "ERROR: cannot copy \"" << src.string()
                  << "\" to \"" << dst.string()
                  << "\": " << ec.message() << "\n";
        return false;
    }
    return true;
}

void process_single_pair_file(const fs::path& refPath,
                              const fs::path& distPath,
                              const CliOptions& opts)
{
    cv::Mat refBGR  = cv::imread(refPath.string(),  cv::IMREAD_COLOR);
    cv::Mat distBGR = cv::imread(distPath.string(), cv::IMREAD_COLOR);

    if (refBGR.empty()) {
        std::cerr << "Cannot read ref image: " << refPath << "\n";
        return;
    }
    if (distBGR.empty()) {
        std::cerr << "Cannot read dist image: " << distPath << "\n";
        return;
    }

    if (refBGR.size() != distBGR.size()) {
        std::cerr << "Size mismatch: " << refPath << " vs " << distPath << "\n";
        return;
    }

    double mse = compute_mse(refBGR, distBGR);

    std::cout << refPath << " " << distPath << " : mse=" << mse
            << "rmse=" << sqrt(mse) << "\n";
}

void process_directory_mode(const CliOptions& opts)
{
    fs::path refDir  = opts.ref;
    fs::path distDir = opts.dist;

    if (!fs::is_directory(refDir)) {
        std::cerr << "Ref is not a directory: " << refDir << "\n";
        return;
    }
    if (!fs::is_directory(distDir)) {
        std::cerr << "Dist is not a directory: " << distDir << "\n";
        return;
    }

    fs::path csvPath = "mse_info.csv";
    std::ofstream csv(csvPath);
    if (!csv.is_open()) {
        std::cerr << "ERROR: cannot write CSV: " << csvPath << "\n";
        return;
    }
    int csvFlushCounter = 0;

    // Loading the list of refs
    auto refFiles  = collect_reference_files(refDir);
    auto distFiles = collect_distorted_files(distDir);
    auto groups    = group_distorted_by_reference(refFiles, distFiles);

    const std::size_t totalRefs = refFiles.size();
    for (std::size_t i = 0; i < totalRefs; ++i) {
        const fs::path& refPath = refFiles[i];
        std::string refKey = stem_lower(refPath); // klucz w mapie (lowercase stem)

        auto it = groups.find(refKey);
        if (it == groups.end() || it->second.empty()) {
            std::cout << "[ref " << (i + 1) << "/" << totalRefs << "] "
                      << refPath << " : no matching distorted files\n";
            continue;
        }

        const auto& distForThisRef = it->second;

        std::cout << "[ref " << (i + 1) << "/" << totalRefs << "] "
                  << refPath << " : " << distForThisRef.size()
                  << " distorted files\n";

        for (const auto& distPath : distForThisRef) {
            cv::Mat refBGR  = cv::imread(refPath.string(),  cv::IMREAD_COLOR);
            cv::Mat distBGR = cv::imread(distPath.string(), cv::IMREAD_COLOR);

            if (refBGR.empty() || distBGR.empty()) {
                std::cerr << "ERROR reading pair: " << refPath
                          << " vs " << distPath << "\n";
                continue;
            }
            if (refBGR.size() != distBGR.size()) {
                std::cerr << "Size mismatch: " << refPath
                          << " vs " << distPath << "\n";
                continue;
            }

            auto mse = compute_mse(refBGR, distBGR);

            // CSV: dist filename
            csv << distPath.filename().string() << "," << mse
                                                << "," << sqrt(mse) << "\n";
            csvFlushCounter++;
            if (csvFlushCounter >= 20) {
                csv.flush();
                csvFlushCounter = 0;
            }

            std::cout << "  " << distPath.filename()
                      << " mse=" << mse << "rmse=" << sqrt(mse) << "\n";
        }
    }
}

int main(int argc, char** argv)
{
    CliOptions opts;
    if (!parse_args(argc, argv, opts)) {
        return 1;
    }

    bool refIsFile  = fs::is_regular_file(opts.ref);
    bool distIsFile = fs::is_regular_file(opts.dist);

    bool refIsDir   = fs::is_directory(opts.ref);
    bool distIsDir  = fs::is_directory(opts.dist);

    if (refIsFile && distIsFile) {
        // single file mode: ref, dist, out-file
        process_single_pair_file(opts.ref, opts.dist, opts);
    } else if (refIsDir && distIsDir) {
        // directory mode: ref-dir, dist-dir, out-dir
        auto start = std::chrono::high_resolution_clock::now();
        process_directory_mode(opts);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duration: " << duration.count()/1e3 << " s.\n";
    } else {
        std::cerr << "ERROR: either all three paths must be files (ref, dist, out_file),\n"
                  << "or ref/dist must be directories and out must be a directory.\n";
        return 1;
    }

    return 0;
}
