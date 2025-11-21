#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "iqalab/impulse.hpp"
#include "iqalab/utils/mask_utils.hpp"

namespace fs = std::filesystem;

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

static cv::Mat load_or_fail(const fs::path& path)
{
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: cannot read image: " << path.string() << "\n";
    }
    return img;
}

static bool process_pair(const fs::path& refPath,
                         const fs::path& distPath,
                         const fs::path& outPath,
                         const fs::path& outMaskPath)
{
    cv::Mat refBGR  = load_or_fail(refPath);
    cv::Mat distBGR = load_or_fail(distPath);
    if (refBGR.empty() || distBGR.empty()) {
        return false;
    }
    if (refBGR.size() != distBGR.size()) {
        std::cerr << "ERROR: size mismatch between "
                  << refPath.string() << " and " << distPath.string() << "\n";
        return false;
    }

    cv::Mat mask = iqa::make_channel_max_diff_mask(refBGR, distBGR);
    size_t count = iqa::count_nonzero_threshold(mask,1);
    std::cout << "Processing mask: " << distPath.filename().string()
                << "  -> impulses detected = " << count << "\n";

    if (!cv::imwrite(outMaskPath.string(), mask)) {
        std::cerr << "ERROR: cannot write output image: " << outMaskPath.string() << "\n";
        return false;
    }

    cv::Mat cleanedBGR;
    iqa::ImpulseStats stats = iqa::clean_impulse_image(refBGR, distBGR, cleanedBGR);

    std::cout << "Processing pair: " << distPath.filename().string()
              << "  -> impulses detected = " << stats.count << "\n";

    if (!cv::imwrite(outPath.string(), cleanedBGR)) {
        std::cerr << "ERROR: cannot write output image: " << outPath.string() << "\n";
        return false;
    }

    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: tid_impulse <TID_root_directory>\n";
        return 1;
    }

    fs::path root = fs::path(argv[1]);

    // Input paths in the TID root.
    fs::path refPath       = root / "reference_images" / "I01.BMP";
    fs::path impulsePath   = root / "distorted_images" / "i01_06_1.bmp";
    fs::path noimpulsePath = root / "distorted_images" / "i01_01_5.bmp";

    // Local copies in current working directory.
    fs::path refCopy       = "ref.bmp";
    fs::path impulseCopy   = "impulse.bmp";
    fs::path noimpulseCopy = "noimpulse.bmp";

    if (!copy_or_fail(refPath,       refCopy))       return 1;
    if (!copy_or_fail(impulsePath,   impulseCopy))   return 1;
    if (!copy_or_fail(noimpulsePath, noimpulseCopy)) return 1;

    // Generate cleaned images.
    fs::path impulseCleaned   = "impulse_cleaned.bmp";
    fs::path impulseMask   = "impulse_mask.png";
    fs::path noImpulseCleaned = "noimpulse_cleaned.bmp";
    fs::path noImpulseMask   = "noimpulse_mask.png";

    if (!process_pair(refCopy, impulseCopy, impulseCleaned, impulseMask)) {
        return 1;
    }
    if (!process_pair(refCopy, noimpulseCopy, noImpulseCleaned, noImpulseMask)) {
        return 1;
    }

    return 0;
}
