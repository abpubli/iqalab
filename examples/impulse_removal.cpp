#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "iqalab/impulse.hpp"

namespace fs = std::filesystem;
using namespace iqa;

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
        std::cerr << "  impulse_removal_demo <ref_file> <dist_file> <out_file> [--threshold N] [--dry]\n";
        std::cerr << "  impulse_removal_demo <ref_dir>  <dist_dir>  <out_dir>  [--threshold N] [--dry]\n";
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

bool is_image_file(const fs::path& p)
{
    if (!fs::is_regular_file(p)) return false;

    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".avif"
    };

    std::string ext = p.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    for (const auto& e : exts) {
        if (ext == e) return true;
    }
    return false;
}

std::string to_lower(const std::string& s)
{
    std::string r = s;
    for (char& c : r) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return r;
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
    fs::path csvPath = outDir / "impulses.csv";
    std::ofstream csv(csvPath);
    if (!csv.is_open()) {
        std::cerr << "ERROR: cannot write CSV: " << csvPath << "\n";
        return;
    }
    int csvFlushCounter = 0;

    // Loading the list of refs
    std::vector<fs::path> refFiles;
    for (auto& e : fs::directory_iterator(refDir)) {
        if (is_image_file(e.path())) {
            refFiles.push_back(e.path());
        }
    }
    std::sort(refFiles.begin(), refFiles.end());

    // We load the list of dists and group them by stem prefix (ref)
    std::vector<fs::path> distFiles;
    for (auto& e : fs::directory_iterator(distDir)) {
        if (is_image_file(e.path())) {
            distFiles.push_back(e.path());
        }
    }
    std::sort(distFiles.begin(), distFiles.end());

    // Mapping: lower(stem(ref)) -> list of matching dist
    std::unordered_map<std::string, std::vector<fs::path>> groups;

    for (const auto& dist : distFiles) {
        std::string distStemLower = to_lower(dist.stem().string());
        // matching will be done later by starts_with,
        // so for now we are just collecting the list
        // (you can also build it live, but this way is simpler + clearer).
    }

    // TID-style grouping:
    // for each ref, we search for all dists whose stem (lowercase)
    // begins with stem(ref) (lowercase), e.g.:
    // ref: I01.BMP   -> stem “I01” -> “i01”
    // dist: i01_01_1.bmp, i01_01_2.bmp, ...

    const std::size_t totalRefs = refFiles.size();
    for (std::size_t i = 0; i < totalRefs; ++i) {
        const fs::path& refPath = refFiles[i];
        std::string refStemLower = to_lower(refPath.stem().string());

        // collect all dists with this prefix
        std::vector<fs::path> distForThisRef;
        for (const auto& dist : distFiles) {
            std::string distStemLower = to_lower(dist.stem().string());
            if (distStemLower.rfind(refStemLower, 0) == 0) { // starts_with
                distForThisRef.push_back(dist);
            }
        }

        if (distForThisRef.empty()) {
            std::cout << "[ref " << (i + 1) << "/" << totalRefs << "] "
                      << refPath << " : no matching distorted files\n";
            continue;
        }

        std::cout << "[ref " << (i + 1) << "/" << totalRefs << "] "
                  << refPath << " : " << distForThisRef.size()
                  << " distorted files\n";

        // Dist for this ref are already sorted globally, so
        // we keep the alphabetical order.

        for (const auto& distPath : distForThisRef)
        {
            fs::path outPath = make_output_path_for_dist(outDir, distPath, "_impulses");

            cv::Mat refBGR  = cv::imread(refPath.string(),  cv::IMREAD_COLOR);
            cv::Mat distBGR = cv::imread(distPath.string(), cv::IMREAD_COLOR);

            if (refBGR.empty() || distBGR.empty() || refBGR.size() != distBGR.size()) {
                std::cerr << "ERROR reading pair: " << refPath << " vs " << distPath << "\n";
                continue;
            }

            cv::Mat cleaned;
            ImpulseStats stats = clean_impulse_image(refBGR, distBGR, cleaned);
            int impulses = stats.count;

            // --- CSV LOG ---
            csv << distPath.filename().string() << "," << impulses << "\n";
            csvFlushCounter++;
            if (csvFlushCounter >= 20) {
                csv.flush();
                csvFlushCounter = 0;
            }

            // --- Save output ---
            if (!opts.dry && impulses >= opts.threshold) {
                fs::create_directories(outPath.parent_path());
                if (!cv::imwrite(outPath.string(), cleaned)) {
                    std::cerr << "Failed to write: " << outPath << "\n";
                }
            }

            std::cout << "  " << distPath.filename() << " impulses=" << impulses << "\n";
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
    bool outIsFile  = fs::path(opts.out).has_extension();

    bool refIsDir   = fs::is_directory(opts.ref);
    bool distIsDir  = fs::is_directory(opts.dist);
    bool outIsDir   = fs::is_directory(opts.out) ||
                      (!outIsFile && !fs::exists(opts.out));

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
