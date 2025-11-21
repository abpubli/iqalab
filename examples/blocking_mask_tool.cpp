#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "iqalab/image_type.hpp"          // stem_lower, is_supported_image
#include "iqalab/utils/file_grouping.hpp" // collect_reference_files, collect_distorted_files, group_distorted_by_reference
#include "iqalab/utils/path_utils.hpp"    // path_to_utf8, ...

namespace fs = std::filesystem;
using namespace iqa;
using namespace iqa::utils;

// Zakładamy, że implementacja jest gdzieś indziej (np. w blocking.cpp / iqalab.hpp).
namespace iqa {
    cv::Mat blocking_to_mask(const cv::Mat& refRgb, const cv::Mat& distRgb);
}

struct Options {
    fs::path refPath;
    fs::path distPath;
    fs::path outPath;
};

//----------------------------------------------------------------------
// Pomocnicze

static void print_usage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <ref_image> <dist_image> <out_image>\n"
        << "  " << argv0 << " <ref_dir>   <dist_dir>   <out_dir>\n";
}

static bool parse_args(int argc, char** argv, Options& opts)
{
    if (argc != 4) {
        print_usage(argv[0]);
        return false;
    }
    opts.refPath  = fs::path(argv[1]);
    opts.distPath = fs::path(argv[2]);
    opts.outPath  = fs::path(argv[3]);
    return true;
}

static cv::Mat safe_imread_color(const fs::path& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: cannot read image: " << path << "\n";
    }
    return img;
}

// Nakładanie maski: >0 -> kolor czarny, 0 -> bez zmian
static cv::Mat apply_block_mask(const cv::Mat& distRgb, const cv::Mat& mask)
{
    if (distRgb.empty() || mask.empty()) {
        return cv::Mat();
    }

    if (distRgb.rows != mask.rows || distRgb.cols != mask.cols) {
        std::cerr << "ERROR: mask size mismatch ("
                  << mask.cols << "x" << mask.rows << ") vs image ("
                  << distRgb.cols << "x" << distRgb.rows << ")\n";
        return cv::Mat();
    }

    if (distRgb.type() != CV_8UC3) {
        std::cerr << "ERROR: distRgb must be CV_8UC3 (3-channel 8-bit BGR)\n";
        return cv::Mat();
    }

    if (mask.type() != CV_8UC1) {
        std::cerr << "ERROR: mask must be CV_8UC1 (single-channel 8-bit)\n";
        return cv::Mat();
    }

    cv::Mat out = distRgb.clone();

    const int rows = distRgb.rows;
    const int cols = distRgb.cols;
    for (int y = 0; y < rows; ++y) {
        const std::uint8_t* mrow = mask.ptr<std::uint8_t>(y);
        cv::Vec3b* orow          = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] != 0) {
                orow[x] = cv::Vec3b(0, 0, 0); // czarny piksel
            }
        }
    }

    return out;
}

//----------------------------------------------------------------------
// Tryb: pojedyncze pliki

static void process_single_file(const Options& opts)
{
    if (!fs::is_regular_file(opts.refPath)) {
        std::cerr << "Ref is not a file: " << opts.refPath << "\n";
        return;
    }
    if (!fs::is_regular_file(opts.distPath)) {
        std::cerr << "Dist is not a file: " << opts.distPath << "\n";
        return;
    }

    cv::Mat refImg  = safe_imread_color(opts.refPath);
    cv::Mat distImg = safe_imread_color(opts.distPath);
    if (refImg.empty() || distImg.empty()) {
        return;
    }

    cv::Mat mask = blocking_to_mask(refImg, distImg);
    if (mask.empty()) {
        std::cerr << "ERROR: blocking_to_mask returned empty mask\n";
        return;
    }

    cv::Mat out = apply_block_mask(distImg, mask);
    if (out.empty()) {
        return;
    }

    fs::path outPath = opts.outPath;
    if (fs::is_directory(outPath)) {
        // Jeśli trzeci argument jest katalogiem, zbuduj nazwę: stem(dist) + "_blocks" + ext(dist).
        outPath = outPath / (opts.distPath.stem().string() + "_blocks" + opts.distPath.extension().string());
    }

    if (!cv::imwrite(outPath, out)) {
        std::cerr << "ERROR: cannot write image: " << outPath << "\n";
    } else {
        std::cout << "Wrote: " << outPath << "\n";
    }
}

//----------------------------------------------------------------------
// Tryb: katalogi

static void process_directory_mode(const Options& opts)
{
    const fs::path& refDir  = opts.refPath;
    const fs::path& distDir = opts.distPath;
    const fs::path& outDir  = opts.outPath;

    if (!fs::is_directory(refDir)) {
        std::cerr << "Ref is not a directory: " << refDir << "\n";
        return;
    }
    if (!fs::is_directory(distDir)) {
        std::cerr << "Dist is not a directory: " << distDir << "\n";
        return;
    }

    fs::create_directories(outDir);

    auto refFiles  = collect_reference_files(refDir);
    auto distFiles = collect_distorted_files(distDir);
    auto groups    = group_distorted_by_reference(refFiles, distFiles);

    const std::size_t totalRefs = refFiles.size();
    if (totalRefs == 0) {
        std::cerr << "No reference images found in: " << refDir << "\n";
        return;
    }

    std::cout << "Found " << totalRefs << " reference images\n";

    for (std::size_t i = 0; i < totalRefs; ++i) {
        const fs::path& refPath = refFiles[i];
        const std::string refKey = stem_lower(refPath); // klucz jak w impulse_mask_tool

        auto it = groups.find(refKey);
        if (it == groups.end() || it->second.empty()) {
            std::cout << "[ref] " << refPath << " -> no distorted images\n";
            continue;
        }

        std::cout << "[ref] " << refPath << " -> " << it->second.size() << " distorted\n";

        cv::Mat refImg = safe_imread_color(refPath);
        if (refImg.empty()) {
            std::cerr << "Skipping ref (cannot read): " << refPath << "\n";
            continue;
        }

        const auto& distList = it->second;
        for (const fs::path& distPath : distList) {
            if (!is_image_file(distPath)) {
                std::cout << "Skipping unsupported image: " << distPath << "\n";
                continue;
            }

            cv::Mat distImg = safe_imread_color(distPath);
            if (distImg.empty()) {
                std::cerr << "Skipping dist (cannot read): " << distPath << "\n";
                continue;
            }

            cv::Mat mask = blocking_to_mask(refImg, distImg);
            if (mask.empty()) {
                std::cerr << "blocking_to_mask returned empty mask for: " << distPath << "\n";
                continue;
            }

            cv::Mat out = apply_block_mask(distImg, mask);
            if (out.empty()) {
                std::cerr << "apply_block_mask failed for: " << distPath << "\n";
                continue;
            }

            fs::path outPath = outDir / (distPath.stem().string() + "_blocks" + distPath.extension().string());

            if (!cv::imwrite(outPath, out)) {
                std::cerr << "ERROR: cannot write image: " << outPath << "\n";
            } else {
                std::cout << "Wrote: " << outPath << "\n";
            }
        }
    }
}

//----------------------------------------------------------------------

int main(int argc, char** argv)
{
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return 1;
    }

    const bool refIsFile  = fs::is_regular_file(opts.refPath);
    const bool distIsFile = fs::is_regular_file(opts.distPath);
    const bool outIsFile  = fs::is_regular_file(opts.outPath);

    const bool refIsDir   = fs::is_directory(opts.refPath);
    const bool distIsDir  = fs::is_directory(opts.distPath);
    const bool outIsDir   = fs::is_directory(opts.outPath);

    if (refIsFile && distIsFile && (outIsFile || !outIsDir)) {
        // Tryb: pojedyncze pliki (trzeci argument może być jeszcze nieistniejącym plikiem).
        process_single_file(opts);
    } else if (refIsDir && distIsDir) {
        // Tryb: katalogi (trzeci będzie traktowany jako katalog wyjściowy).
        process_directory_mode(opts);
    } else {
        std::cerr << "ERROR: either all three paths must be files (ref, dist, out_image),\n"
                  << "or ref/dist must be directories and out must be a directory.\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
