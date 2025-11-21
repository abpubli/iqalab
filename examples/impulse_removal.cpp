#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "iqalab/impulse.hpp"
#include "iqalab/utils/file_grouping.hpp"
#include "iqalab/utils/path_utils.hpp"

namespace fs = std::filesystem;
using namespace iqa;
using namespace iqa::utils;

struct CliOptions {
    fs::path ref;
    fs::path dist;
    fs::path out;
    int threshold = 1;    // minimum impulses to save a file
    bool dry = false;     // dry-run: only report, don't save
};

bool parse_args(int argc, char** argv, CliOptions& opts)
{
    if (argc < 4) {
        std::cerr << "Usage:\n";
        std::cerr << "  impulse_removal <ref_file> <dist_file> <out_file> [--threshold N] [--dry]\n";
        std::cerr << "  impulse_removal <ref_dir>  <dist_dir>  <out_dir>  [--threshold N] [--dry]\n";
        return false;
    }

    opts.ref  = argv[1];
    opts.dist = argv[2];
    opts.out  = argv[3];

    int i = 4;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "--threshold" && i + 1 < argc) {
            opts.threshold = std::stoi(argv[i + 1]);
            i += 2;
        } else if (arg == "--dry") {
            opts.dry = true;
            ++i;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            ++i;
        }
    }
    return true;
}

fs::path make_output_path_for_dist(const fs::path& outDir,
                                   const fs::path& distFile,
                                   const std::string& suffix = "_impulses")
{
    fs::path name = distFile.filename(); // "i01_01_1.bmp"
    fs::path stem = name.stem();         // "i01_01_1"
    fs::path ext  = name.extension();    // ".bmp"
    fs::path outName = stem.string() + suffix + ext.string();
    return outDir / outName;
}

void process_single_pair_file(const fs::path& refPath,
                              const fs::path& distPath,
                              const fs::path& outPath,
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

    cv::Mat cleaned;
    ImpulseStats stats = clean_impulse_image(refBGR, distBGR, cleaned);

    std::cout << refPath << " | " << distPath << " : impulses=" << stats.count;

    if (stats.count < opts.threshold) {
        std::cout << " (below threshold, skip)";
    }

    if (opts.dry) {
        std::cout << " [dry-run]\n";
        return;
    }

    if (stats.count < opts.threshold) {
        std::cout << "\n";
        return;
    }

    fs::create_directories(outPath.parent_path());
    if (!cv::imwrite(outPath.string(), cleaned)) {
        std::cerr << "  -> failed to write: " << outPath << "\n";
    } else {
        std::cout << "  -> saved: " << outPath << "\n";
    }
}

void process_directory_mode(const CliOptions& opts)
{
    fs::path refDir  = opts.ref;
    fs::path distDir = opts.dist;
    fs::path outDir  = opts.out;

    if (!fs::is_directory(refDir)) {
        std::cerr << "Ref is not a directory: " << refDir << "\n";
        return;
    }
    if (!fs::is_directory(distDir)) {
        std::cerr << "Dist is not a directory: " << distDir << "\n";
        return;
    }

    fs::create_directories(outDir);

    // CSV output in outDir
    fs::path csvPath = outDir / "impulses.csv";
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
            fs::path outPath = make_output_path_for_dist(outDir, distPath, "_impulses");

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

            cv::Mat cleaned;
            ImpulseStats stats = clean_impulse_image(refBGR, distBGR, cleaned);
            int impulses = stats.count;

            // CSV: dist filename + impulses
            csv << distPath.filename().string() << "," << impulses << "\n";
            csvFlushCounter++;
            if (csvFlushCounter >= 20) {
                csv.flush();
                csvFlushCounter = 0;
            }

            // Save image if not dry-run and above threshold
            if (!opts.dry && impulses >= opts.threshold) {
                fs::create_directories(outPath.parent_path());
                if (!cv::imwrite(outPath.string(), cleaned)) {
                    std::cerr << "Failed to write: " << outPath << "\n";
                }
            }

            std::cout << "  " << distPath.filename()
                      << " impulses=" << impulses;
            if (impulses < opts.threshold) {
                std::cout << " (below threshold, skip)";
            }
            if (opts.dry) {
                std::cout << " [dry-run]";
            }
            std::cout << "\n";
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
        process_single_pair_file(opts.ref, opts.dist, opts.out, opts);
    } else if (refIsDir && distIsDir) {
        // directory mode: ref-dir, dist-dir, out-dir
        process_directory_mode(opts);
    } else {
        std::cerr << "ERROR: either all three paths must be files (ref, dist, out_file),\n"
                  << "or ref/dist must be directories and out must be a directory.\n";
        return 1;
    }

    return 0;
}
