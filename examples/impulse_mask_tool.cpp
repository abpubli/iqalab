#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "iqalab/image_type.hpp"          // stem_lower, is_supported_image
#include "iqalab/impulse.hpp"             // impulse_to_mask(...)
#include "iqalab/utils/file_grouping.hpp" // collect_reference_files, ...
#include "iqalab/utils/path_utils.hpp"

namespace fs = std::filesystem;
using namespace iqa;
using namespace iqa::utils;

struct CliOptions {
    fs::path ref;
    fs::path dist;
    fs::path out;
};

static bool parse_args(int argc, char** argv, CliOptions& opts)
{
    if (argc < 4) {
        std::cerr << "Usage:\n";
        std::cerr << "  impulse_mask_tool <ref_file> <dist_file> <out_mask_png>\n";
        std::cerr << "  impulse_mask_tool <ref_dir>  <dist_dir>  <out_dir>\n";
        return false;
    }

    opts.ref  = argv[1];
    opts.dist = argv[2];
    opts.out  = argv[3];
    return true;
}

// Output mask path for a single distorted file.
// Example:
//   dist: i01_01_1.bmp
//   outDir: /tmp/masks
//   suffix: "_impulse_mask"
// Result:
//   /tmp/masks/i01_01_1_impulse_mask.png
static fs::path make_mask_output_path(const fs::path& outDir,
                                      const fs::path& distFile,
                                      const std::string& suffix = "_impulse_mask")
{
    fs::path name = distFile.filename(); // "i01_01_1.bmp"
    fs::path stem = name.stem();         // "i01_01_1"
    fs::path outName = stem.string() + suffix + ".png";
    return outDir / outName;
}

// Single (ref, dist, out_mask) mode.
static void process_single_file(const CliOptions& opts)
{
    const fs::path& refPath  = opts.ref;
    const fs::path& distPath = opts.dist;
    const fs::path& outPath  = opts.out;

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

    size_t nImp;
    cv::Mat mask = impulse_to_mask_bgr8(refBGR, distBGR, nImp); // CV_8U, 0/255

    auto parentPath = outPath.parent_path();
    if (!parentPath.empty())
        fs::create_directories(outPath.parent_path());
    if (!cv::imwrite(outPath.string(), mask)) {
        std::cerr << "Failed to write mask: " << outPath << "\n";
    } else {
        std::cout << "Wrote impulse mask: " << outPath
                    << " -> impulses=" << nImp << "\n";
    }
}

// Directory mode: ref_dir, dist_dir, out_dir
// For each reference image we find all distorted images that match by basename
// (TID-like naming) using utils::group_distorted_by_reference().
// For each (ref, dist) pair we write a PNG mask into out_dir.
static void process_directory_mode(const CliOptions& opts)
{
    const fs::path& refDir  = opts.ref;
    const fs::path& distDir = opts.dist;
    const fs::path& outDir  = opts.out;

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

    for (std::size_t i = 0; i < totalRefs; ++i) {
        const fs::path& refPath = refFiles[i];
        std::string refKey = stem_lower(refPath); // lowercase stem, used as map key

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

        cv::Mat refBGR = cv::imread(refPath.string(), cv::IMREAD_COLOR);
        if (refBGR.empty()) {
            std::cerr << "  ERROR: cannot read ref image: " << refPath << "\n";
            continue;
        }

        for (const auto& distPath : distForThisRef) {
            cv::Mat distBGR = cv::imread(distPath.string(), cv::IMREAD_COLOR);
            if (distBGR.empty()) {
                std::cerr << "  ERROR: cannot read dist image: " << distPath << "\n";
                continue;
            }
            if (distBGR.size() != refBGR.size()) {
                std::cerr << "  Size mismatch: " << refPath
                          << " vs " << distPath << "\n";
                continue;
            }

            std::size_t nImp;
            cv::Mat mask = impulse_to_mask_bgr8(refBGR, distBGR, nImp);
            // count impulses
            fs::path outMaskPath = make_mask_output_path(outDir, distPath);
            fs::create_directories(outMaskPath.parent_path());

            if (!cv::imwrite(outMaskPath.string(), mask)) {
                std::cerr << "  Failed to write mask: " << outMaskPath << "\n";
            } else {
                std::cout << "  " << distPath.filename()
                          << " -> " << outMaskPath.filename()
                          << " -> impulses=" << nImp << "\n";
            }
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
        // Single (ref, dist, out_mask_png)
        process_single_file(opts);
    } else if (refIsDir && distIsDir) {
        // Directory mode: ref_dir, dist_dir, out_dir
        process_directory_mode(opts);
    } else {
        std::cerr << "ERROR: either all three paths must be files (ref, dist, out_mask_png),\n"
                  << "or ref/dist must be directories and out must be a directory.\n";
        if (refIsFile)
            std::cerr << opts.ref << " is file but " << opts.dist << " is a directory.\n";
        else
            std::cerr << opts.ref << " is directory but " << opts.dist << " is a file.\n";
        return 1;
    }

    return 0;
}
