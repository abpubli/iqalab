#include "iqalab/region_provider.hpp"
#include "iqalab/utils/path_utils.hpp"

#include <iqalab/blur.hpp>
#include <iqalab/mse.hpp>
#include <iqalab/region_masks.hpp>
#include <iqalab/utils/file_grouping.hpp>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Simple pair of reference and distorted file paths (as strings for CSV output).
struct Pair
{
    std::string refPath;
    std::string distPath;
};

// Masks used for blur measurement (dilated regions).
struct BlurRegionMasks
{
    cv::Mat flat;
    cv::Mat mid;
    cv::Mat detail;
};

// Load image from disk and convert to Lab (CV_32FC3).
// This helper keeps all color-space conversions in one place.
bool load_image_lab32(const std::string& path, cv::Mat& lab)
{
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        std::cerr << "Cannot read image: " << path << "\n";
        return false;
    }

    cv::Mat lab8;
    cv::cvtColor(bgr, lab8, cv::COLOR_BGR2Lab);
    lab8.convertTo(lab, CV_32FC3);
    return true;
}

// Load pairs from a text file where each line contains:
//   <ref_path> <dist_path>
// separated by whitespace.
std::vector<Pair> load_pairs_from_file(const std::string& listPath)
{
    std::vector<Pair> pairs;
    std::ifstream in(listPath);
    if (!in)
    {
        std::cerr << "Cannot open pairs file: " << listPath << "\n";
        return pairs;
    }

    std::string ref, dist;
    while (in >> ref >> dist)
    {
        pairs.push_back(Pair{ref, dist});
    }
    return pairs;
}

// Build pairs from two directory roots using file grouping utilities.
// This uses iqa::utils::group_distorted_by_reference() to associate
// each reference file with its distorted counterparts.
std::vector<Pair> load_pairs_from_dirs(const std::string& refsRoot,
                                              const std::string& distsRoot)
{
    // in iqalab/utils/file_grouping.hpp and file_grouping.cpp.
    //
    // The idea:
    //   - there is some Group/Record type that contains:
    //       - a reference image path
    //       - a collection of distorted image paths for that reference
    //
    // Example (pseudo-type):
    //   using Group = iqa::utils::DistortedByReferenceGroup;
    //   std::vector<Group> groups =
    //       iqa::utils::group_distorted_by_reference(refsRoot, distsRoot);

    auto refFiles  = iqa::utils::collect_reference_files(refsRoot);
    auto distFiles = iqa::utils::collect_distorted_files(distsRoot);
    auto groups    = iqa::utils::group_distorted_by_reference(refFiles, distFiles);

    const std::size_t totalRefs = refFiles.size();
    if (totalRefs == 0) {
        std::cerr << "No reference images found in: " << refsRoot << "\n";
        exit(1);
    }

    std::vector<Pair> pairs;
    // The following assumes something like:
    //   group.reference_path  -> std::filesystem::path
    //   group.distorted_paths -> std::vector<std::filesystem::path>
    for (std::size_t i = 0; i < totalRefs; ++i) {
        const fs::path& refPath = refFiles[i];
        std::string refKey = iqa::utils::stem_lower(refPath); // lowercase stem, used as map key
        auto it = groups.find(refKey);
        if (it == groups.end() || it->second.empty()) {
            std::cout << "[ref " << (i + 1) << "/" << totalRefs << "] "
                      << refPath << " : no matching distorted files\n";
            continue;
        }
        const auto& distForThisRef = it->second;
        for (const auto& distPath : distForThisRef) {
            pairs.push_back(Pair{refPath, distPath.string()});
        }
    }
    return pairs;
}

// Create dilated masks for blur measurement.
//
// regionMasks.* are original flat/mid/detail masks from iqa::compute_region_masks()
// (0/255, CV_8U). For blur we dilate these masks with configurable radii in pixels.
//
// r_flat, r_mid, r_detail:
//   - dilation radii for each region (0 = no dilation, use original mask).
BlurRegionMasks make_blur_region_masks(const iqa::RegionMasks& regionMasks,
                                       int r_flat,
                                       int r_mid,
                                       int r_detail)
{
    BlurRegionMasks blurMasks;

    auto dilate_if_needed = [](const cv::Mat& srcMask, int r) -> cv::Mat
    {
        CV_Assert(srcMask.type() == CV_8U);
        if (r <= 0)
        {
            // No dilation required, just return a copy.
            return srcMask.clone();
        }

        const int k = 2 * r + 1;
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(k, k)
        );

        cv::Mat dst;
        cv::dilate(srcMask, dst, kernel);
        return dst;
    };

    blurMasks.flat   = dilate_if_needed(regionMasks.flat,   r_flat);
    blurMasks.mid    = dilate_if_needed(regionMasks.mid,    r_mid);
    blurMasks.detail = dilate_if_needed(regionMasks.detail, r_detail);

    return blurMasks;
}

// Print usage information for this CLI tool.
void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0 << " ref.png dist.png\n";
    std::cerr << "  " << argv0 << " pairs.txt\n";
    std::cerr << "  " << argv0 << " --dirs <refs_root> <dists_root>\n";
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<Pair> pairs;

    // Mode 1: single pair of images: ref.png dist.png

    // Heuristic: if second argument is not "--dirs", we treat the two arguments
    // as a single reference/distorted pair OR a pairs.txt file.
    // Here we decide:
    //
    // - if there is an extension ".txt" on argv[1] -> treat as pairs file
    // - else -> treat as single ref/dist pair
    std::string arg1 = argv[1];
    std::string arg2 = argv[2];

    if (arg1.size() >= 4 &&
        (arg1.compare(arg1.size() - 4, 4, ".txt") == 0 ||
         arg1.compare(arg1.size() - 4, 4, ".lst") == 0))
    {
        // pairs.txt mode
        pairs = load_pairs_from_file(arg1);
        if (pairs.empty())
        {
            std::cerr << "No pairs loaded from file: " << arg1 << "\n";
            return 1;
        }
    }
    else
    {
        bool refIsFile  = fs::is_regular_file(arg1);
        bool distIsFile = fs::is_regular_file(arg2);

        bool refIsDir   = fs::is_directory(arg1);
        bool distIsDir  = fs::is_directory(arg2);

        if (refIsFile && distIsFile) {
            pairs.push_back(Pair{arg1, arg2});
        } else if (refIsDir && distIsDir) {
            std::string refsRoot  = arg1;
            std::string distsRoot = arg2;
            pairs = load_pairs_from_dirs(refsRoot, distsRoot);
            if (pairs.empty())
            {
                std::cerr << "No pairs built from directories: "
                          << refsRoot << " and " << distsRoot << "\n";
                return 1;
            }
        } else {
            std::cerr << "ERROR: either all three paths must be files (ref, dist, out_mask_png),\n"
                      << "or ref/dist must be directories and out must be a directory.\n";
            if (refIsFile)
                std::cerr << arg1 << " is file but " << arg2 << " is a directory.\n";
            else
                std::cerr << arg1 << " is directory but " << arg2 << " is a file.\n";

            return 1;
        }
    }

    auto regionProvider = iqa::make_default_region_provider();
    // CSV header.
    //
    // We report:
    // - per-region pixel counts for original masks (region_*),
    // - per-region pixel counts for dilated masks used for blur (blur_*),
    // - blur in L and in a+b per region,
    // - MSE in L and combined MSE in a+b (MSE_ab) globally and per region.
    std::cout
        << "ref_path,dist_path,"
        << "n_region_flat,n_region_mid,n_region_detail,"
        << "n_blur_flat,n_blur_mid,n_blur_detail,"
        << "blur_L_flat,blur_L_mid,blur_L_detail,"
        << "blur_ab_flat,blur_ab_mid,blur_ab_detail,"
        << "mse_L_all,mse_ab_all,"
        << "mse_L_flat,mse_L_mid,mse_L_detail,"
        << "mse_ab_flat,mse_ab_mid,mse_ab_detail"
        << "\n";

    for (const auto& p : pairs)
    {
        cv::Mat labRef, labDist;
        if (!load_image_lab32(p.refPath, labRef) ||
            !load_image_lab32(p.distPath, labDist))
        {
            // If either image cannot be loaded, skip this pair.
            continue;
        }

        if (labRef.size() != labDist.size())
        {
            std::cerr << "Size mismatch: " << p.refPath
                      << " vs " << p.distPath << "\n";
            continue;
        }

        // Compute region masks on the reference image (Lab space).
        // This uses your existing region mask computation from iqa.
        iqa::RegionMasks regionMasks = regionProvider->compute_regions(labRef);

        // Pixel counts for original regions.
        double n_region_flat   = static_cast<double>(cv::countNonZero(regionMasks.flat));
        double n_region_mid    = static_cast<double>(cv::countNonZero(regionMasks.mid));
        double n_region_detail = static_cast<double>(cv::countNonZero(regionMasks.detail));

        // Create dilated masks for blur. Radii can be tuned:
        // - r_flat   = 1: slight dilation around flat areas
        // - r_mid    = 2: medium dilation for mid-texture
        // - r_detail = 3: more dilation around strong edges / details
        const int r_flat   = 1;
        const int r_mid    = 2;
        const int r_detail = 3;

        BlurRegionMasks blurMasks = make_blur_region_masks(regionMasks,
                                                           r_flat,
                                                           r_mid,
                                                           r_detail);

        // Pixel counts for blur masks.
        double n_blur_flat   = static_cast<double>(cv::countNonZero(blurMasks.flat));
        double n_blur_mid    = static_cast<double>(cv::countNonZero(blurMasks.mid));
        double n_blur_detail = static_cast<double>(cv::countNonZero(blurMasks.detail));

        // Relative blur in L channel per region (using dilated masks).
        double blur_L_flat   = iqa::blur::relative_blur_L(labRef, labDist, blurMasks.flat);
        double blur_L_mid    = iqa::blur::relative_blur_L(labRef, labDist, blurMasks.mid);
        double blur_L_detail = iqa::blur::relative_blur_L(labRef, labDist, blurMasks.detail);

        // Relative blur in a+b (chroma) per region (using dilated masks).
        double blur_ab_flat   = iqa::blur::relative_blur_ab(labRef, labDist, blurMasks.flat);
        double blur_ab_mid    = iqa::blur::relative_blur_ab(labRef, labDist, blurMasks.mid);
        double blur_ab_detail = iqa::blur::relative_blur_ab(labRef, labDist, blurMasks.detail);

        // Global MSE in Lab channels (no mask).
        double mse_L_all = iqa::mse::lab_channel_mse(labRef, labDist, 0);
        double mse_a_all = iqa::mse::lab_channel_mse(labRef, labDist, 1);
        double mse_b_all = iqa::mse::lab_channel_mse(labRef, labDist, 2);

        // Combined chroma MSE: MSE_ab = MSE_a + MSE_b
        double mse_ab_all = mse_a_all + mse_b_all;

        // Per-region MSE in Lab channels (using original, non-dilated masks).
        double mse_L_flat   = iqa::mse::lab_channel_mse(labRef, labDist, 0, regionMasks.flat);
        double mse_L_mid    = iqa::mse::lab_channel_mse(labRef, labDist, 0, regionMasks.mid);
        double mse_L_detail = iqa::mse::lab_channel_mse(labRef, labDist, 0, regionMasks.detail);

        double mse_a_flat   = iqa::mse::lab_channel_mse(labRef, labDist, 1, regionMasks.flat);
        double mse_a_mid    = iqa::mse::lab_channel_mse(labRef, labDist, 1, regionMasks.mid);
        double mse_a_detail = iqa::mse::lab_channel_mse(labRef, labDist, 1, regionMasks.detail);

        double mse_b_flat   = iqa::mse::lab_channel_mse(labRef, labDist, 2, regionMasks.flat);
        double mse_b_mid    = iqa::mse::lab_channel_mse(labRef, labDist, 2, regionMasks.mid);
        double mse_b_detail = iqa::mse::lab_channel_mse(labRef, labDist, 2, regionMasks.detail);

        double mse_ab_flat   = mse_a_flat   + mse_b_flat;
        double mse_ab_mid    = mse_a_mid    + mse_b_mid;
        double mse_ab_detail = mse_a_detail + mse_b_detail;

        // Output one CSV row per (ref, dist) pair.
        std::cout
            << p.refPath << "," << p.distPath << ","
            << n_region_flat << "," << n_region_mid << "," << n_region_detail << ","
            << n_blur_flat << "," << n_blur_mid << "," << n_blur_detail << ","
            << blur_L_flat << "," << blur_L_mid << "," << blur_L_detail << ","
            << blur_ab_flat << "," << blur_ab_mid << "," << blur_ab_detail << ","
            << mse_L_all << "," << mse_ab_all << ","
            << mse_L_flat << "," << mse_L_mid << "," << mse_L_detail << ","
            << mse_ab_flat << "," << mse_ab_mid << "," << mse_ab_detail
            << "\n";
    }

    return 0;
}
