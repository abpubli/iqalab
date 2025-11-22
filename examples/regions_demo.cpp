#include "iqalab/color.hpp"
#include "iqalab/image_type.hpp"
#include "iqalab/mse.hpp"
#include "iqalab/region_provider.hpp"
#include <iqalab/visualize_regions.hpp>

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

#include <iostream>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iqalab/region_blocks.hpp>   // BlockGrid16, BlockRegionMasks, ...
#include <iqalab/region_provider.hpp> // or wherever RegionProvider / RegionMasks live
#include <iqalab/color.hpp>   // bgr8_to_lab32f
#include <iqalab/visualize_regions.hpp>

namespace fs = std::filesystem;

void process_single_file(
    const fs::path& inPath,
    const fs::path& outPath,
    const iqa::RegionProvider& rp)
{
    // 1) Load input image.
    cv::Mat bgr = cv::imread(inPath.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Cannot read image: " << inPath << std::endl;
        return;
    }

    // 2) Convert to Lab (float) for region provider.
    cv::Mat labRef;
    iqa::bgr8_to_lab32f(bgr, labRef);

    // 3) Compute pixel-level region masks.
    iqa::RegionMasks masks = rp.compute_regions(labRef);

    // Sanity checks: all masks must match image size.
    CV_Assert(masks.flat.size()   == bgr.size());
    CV_Assert(masks.mid.size()    == bgr.size());
    CV_Assert(masks.detail.size() == bgr.size());
    CV_Assert(masks.flat.type()   == CV_8UC1);
    CV_Assert(masks.mid.type()    == CV_8UC1);
    CV_Assert(masks.detail.type() == CV_8UC1);

    // 4) Visualize pixel-level regions (existing behaviour).
    cv::Mat visPixel = visualize_regions(bgr, masks);

    // 5) Build block-level masks (16x16) from pixel masks.
    //
    // NOTE: RegionMasks order is:
    //   - masks.flat
    //   - masks.detail
    //   - masks.mid
    //
    // Our block-building function expects:
    //   (flatMask, midMask, detailMask)
    //
    // so we pass masks.mid as "mid".
    using namespace iqa::regions;

    const int blockSize = 16;
    BlockGrid16 grid = make_block16_grid(bgr.size(), blockSize);

    BlockRegionMasks blockMasks =
        make_block_region_masks_from_pixel_masks(
            grid,
            masks.flat,    // flat
            masks.mid,     // mid
            masks.detail,  // detail
            0.5,           // minDominantFrac
            0.3            // strongPairFrac: flat+detail large, mid small -> classify as mid
        );

    // 6) Visualize block-level classification.
    //
    // Here we draw coloured rectangles per block, based on which block mask is set.
    cv::Mat visBlock = bgr.clone();

    for (int by = 0; by < grid.blocksY; ++by) {
        for (int bx = 0; bx < grid.blocksX; ++bx) {
            const int blockIdx = by * grid.blocksX + bx;
            const cv::Rect r = block_rect(grid, blockIdx);
            if (r.width <= 0 || r.height <= 0)
                continue;

            // Because make_block_region_masks_from_pixel_masks() fills whole block
            // uniformly with 255 in exactly one of the masks (or leaves all zero),
            // we can check a single pixel in the ROI to infer the class.
            cv::Mat1b flatROI   = blockMasks.flat(r);
            cv::Mat1b midROI    = blockMasks.mid(r);
            cv::Mat1b detailROI = blockMasks.detail(r);

            const bool isFlat   = (flatROI.at<uchar>(0, 0)   == 255);
            const bool isMid    = (midROI.at<uchar>(0, 0)    == 255);
            const bool isDetail = (detailROI.at<uchar>(0, 0) == 255);

            cv::Scalar color;
            if (isFlat) {
                // Blue for flat.
                color = cv::Scalar(255, 0, 0);
            } else if (isMid) {
                // Yellow for mid.
                color = cv::Scalar(0, 255, 255);
            } else if (isDetail) {
                // Red for detail.
                color = cv::Scalar(0, 0, 255);
            } else {
                // Gray for unclassified blocks.
                color = cv::Scalar(128, 128, 128);
            }

            cv::rectangle(visBlock, r, color, 1);
        }
    }

    // 7) Save outputs.
    //
    // Pixel-level visualization = original outPath.
    // Block-level visualization = outPath with "_block" suffix.
    fs::path outPixel = outPath;
    fs::path outBlock = outPath;

    outBlock.replace_filename(
        outPath.stem().string() + "_block" + outPath.extension().string()
    );

    if (!cv::imwrite(outPixel.string(), visPixel)) {
        std::cerr << "Cannot write: " << outPixel << std::endl;
    }
    if (!cv::imwrite(outBlock.string(), visBlock)) {
        std::cerr << "Cannot write: " << outBlock << std::endl;
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
