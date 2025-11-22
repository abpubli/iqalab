#include "iqalab/color.hpp"
#include "iqalab/image_type.hpp"
#include "iqalab/mse.hpp"
#include "iqalab/region_provider.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace iqa;

enum class RegionsMode {
    PixelFlatMidDetail,  // existing pixel-based provider (masks)
    Blocks16x16          // new block grid 16x16
};

constexpr RegionsMode kRegionsMode = RegionsMode::Blocks16x16;

struct CliOptions {
    fs::path inPath;
    fs::path outPath;
};

bool parse_args(int argc, char** argv, CliOptions& opts)
{
    if (argc < 3) {
        std::cerr << "Usage:\n";
        std::cerr << "  regions_demo <ref_file> <dist_file>\n";
        std::cerr << "  regions_demo <ref_dir>  <dist_dir>\n";
        return false;
    }

    opts.inPath  = argv[1];
    opts.outPath = argv[2];

    return true;
}

// refL: CV_32F (0..1) – np. L z Lab albo gray
cv::Mat visualize_regions(const cv::Mat& bgr, const iqa::RegionMasks& masks)
{
    CV_Assert(!bgr.empty());
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(masks.flat.size()   == bgr.size());
    CV_Assert(masks.detail.size() == bgr.size());
    CV_Assert(masks.mid.size()    == bgr.size());

    cv::Mat out = bgr.clone();

    const float alphaFlat   = 0.35f;
    const float alphaMid    = 0.35f;
    const float alphaDetail = 0.35f;

    for (int y = 0; y < out.rows; ++y) {
        const uchar* flatRow   = masks.flat.ptr<uchar>(y);
        const uchar* midRow    = masks.mid.ptr<uchar>(y);
        const uchar* detailRow = masks.detail.ptr<uchar>(y);

        cv::Vec3b* prow = out.ptr<cv::Vec3b>(y);

        for (int x = 0; x < out.cols; ++x) {
            cv::Vec3b& p = prow[x];

            // BGR
            float B = static_cast<float>(p[0]);
            float G = static_cast<float>(p[1]);
            float R = static_cast<float>(p[2]);

            if (flatRow[x]) {
                // niebieskawy
                const cv::Vec3f overlay(200.f, 150.f,  50.f); // B,G,R
                B = (1.0f - alphaFlat) * B + alphaFlat * overlay[0];
                G = (1.0f - alphaFlat) * G + alphaFlat * overlay[1];
                R = (1.0f - alphaFlat) * R + alphaFlat * overlay[2];
            }
            if (midRow[x]) {
                // zielonkawy
                const cv::Vec3f overlay( 50.f, 200.f,  50.f);
                B = (1.0f - alphaMid) * B + alphaMid * overlay[0];
                G = (1.0f - alphaMid) * G + alphaMid * overlay[1];
                R = (1.0f - alphaMid) * R + alphaMid * overlay[2];
            }
            if (detailRow[x]) {
                // czerwony
                const cv::Vec3f overlay( 50.f,  50.f, 200.f);
                B = (1.0f - alphaDetail) * B + alphaDetail * overlay[0];
                G = (1.0f - alphaDetail) * G + alphaDetail * overlay[1];
                R = (1.0f - alphaDetail) * R + alphaDetail * overlay[2];
            }

            p[0] = static_cast<uchar>(std::clamp(B, 0.0f, 255.0f));
            p[1] = static_cast<uchar>(std::clamp(G, 0.0f, 255.0f));
            p[2] = static_cast<uchar>(std::clamp(R, 0.0f, 255.0f));
        }
    }
    return out;
}

fs::path make_output_path(const fs::path& outBaseDir,
                          const fs::path& inputFile,
                          const std::string & suffix = "_regions")
{
    fs::path name = inputFile.filename();       // eq. "img01.jpg"
    fs::path stem = name.stem();               // "img01"
    fs::path ext  = name.extension();          // ".jpg"
    fs::path outName = stem.string() + suffix + ext.string();
    return outBaseDir / outName;
}

void process_single_file(const fs::path& inPath, const fs::path& outPath, const iqa::RegionProvider & rp)
{
    cv::Mat bgr = cv::imread(inPath.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Cannot read image: " << inPath << std::endl;
        return;
    }
    cv::Mat labRef;
    iqa::bgr8_to_lab32f(bgr, labRef);
    auto masks = rp.compute_regions(labRef);
    cv::Mat vis = visualize_regions(bgr, masks);
    const fs::path& outDir  = outPath;
    if (!cv::imwrite(outPath.string(), vis)) {
      std::cerr << "Cannot write: " << outPath << std::endl;
    }
}

void process_directory(const fs::path& inDir, const fs::path& outDir, const iqa::RegionProvider & rp) {
    fs::create_directories(outDir);
    // collect a list of images to know N
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(inDir)) {
        if (iqa::is_image_file(e.path())) {
            files.push_back(e.path());
        }
    }
    const size_t total = files.size();
    fs::create_directories(outDir);

    for (size_t i = 0; i < total; ++i) {
        const fs::path& p = files[i];
        // print progress: 5/3000 /path/to/file
        std::cout << (i+1) << "/" << total << " " << p << std::endl;
        fs::path outPath = make_output_path(outDir, p, "_regions");
        process_single_file(p, outPath, rp);
    }
}

int main(int argc, char** argv)
{
    CliOptions opts;
    if (!parse_args(argc, argv, opts)) {
        return 1;
    }
    bool inIsFile  = fs::is_regular_file(opts.inPath);
    bool outIsFile = fs::is_regular_file(opts.outPath);

    bool inIsDir   = fs::is_directory(opts.inPath);
    bool outIsDir  = fs::is_directory(opts.outPath);

    if (!outIsFile && !outIsDir) {
        bool outHasExtension = !opts.outPath.extension().empty();
        if (outHasExtension)
            outIsFile = true;
        else
            outIsDir = true;
    }

    auto regionProvider = iqa::make_default_region_provider();
    if (inIsFile && outIsFile) {
        auto start = std::chrono::high_resolution_clock::now();
        process_single_file(opts.inPath, opts.outPath, *regionProvider);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "duration: " << duration.count()/1e3 << " s.\n";
    } else if (inIsDir && outIsDir) {
        auto start = std::chrono::high_resolution_clock::now();
        process_directory(opts.inPath, opts.outPath, *regionProvider);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "duration: " << duration.count()/1e3 << " s.\n";
    } else {
        std::cerr << "ERROR: either all three paths must be files (ref, dist, out_file),\n"
                  << "or ref/dist must be directories and out must be a directory.\n";
        if (inIsFile)
            std::cerr << opts.inPath << " is file but " << opts.outPath << " is a directory.\n";
        else
            std::cerr << opts.inPath << " is directory but " << opts.outPath << " is a file.\n";
        return 1;
    }

    return 0;
}
