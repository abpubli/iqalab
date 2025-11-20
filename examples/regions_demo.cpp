#include "iqalab/iqalab.hpp"
#include "iqalab/region_masks.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;

fs::path make_output_path(const fs::path& outBaseDir,
                          const fs::path& inputFile,
                          const string& suffix = "_regions")
{
    fs::path name = inputFile.filename();       // eq. "img01.jpg"
    fs::path stem = name.stem();               // "img01"
    fs::path ext  = name.extension();          // ".jpg"
    fs::path outName = stem.string() + suffix + ext.string();
    return outBaseDir / outName;
}

// refL: CV_32F (0..1) – np. L z Lab albo gray
cv::Mat visualize_regions(const cv::Mat& bgr, const iqa::RegionMasks& masks)
{
    CV_Assert(!bgr.empty());
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(masks.flatMask.size()   == bgr.size());
    CV_Assert(masks.detailMask.size() == bgr.size());
    CV_Assert(masks.midMask.size()    == bgr.size());

    cv::Mat out = bgr.clone();

    const float alphaFlat   = 0.35f;
    const float alphaMid    = 0.35f;
    const float alphaDetail = 0.35f;

    for (int y = 0; y < out.rows; ++y) {
        const uchar* flatRow   = masks.flatMask.ptr<uchar>(y);
        const uchar* midRow    = masks.midMask.ptr<uchar>(y);
        const uchar* detailRow = masks.detailMask.ptr<uchar>(y);

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

void process_single_file(const fs::path& inPath, const fs::path& outPathBaseDir)
{
    cv::Mat bgr = cv::imread(inPath.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        cerr << "Cannot read image: " << inPath << endl;
        return;
    }
    iqa::RegionMasks masks = iqa::computeRegionMasks32(bgr);
    cv::Mat vis = visualize_regions(bgr, masks);
    const fs::path& outDir  = outPathBaseDir;
    fs::create_directories(outDir);
    fs::path outPath = make_output_path(outDir, inPath, "_regions");
    if (!cv::imwrite(outPath.string(), vis)) {
        cerr << "Cannot write: " << outPath << endl;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: regions_demo <input_file|input_dir> <output_file|output_dir>\n";
        return 1;
    }

    fs::path inPath  = argv[1];
    fs::path outPath = argv[2];

    if (fs::is_regular_file(inPath)) {
        if (fs::is_directory(outPath)) {
            // output: directory – we generate a name with a suffix
            process_single_file(inPath, outPath);
        } else {
            // output: exact file – we will use it without the suffix
            cv::Mat bgr = cv::imread(inPath.string(), cv::IMREAD_COLOR);
            if (bgr.empty()) {
                cerr << "Cannot read image: " << inPath << endl;
                return 1;
            }
            cv::Mat gray8, refL;
            cv::cvtColor(bgr, gray8, cv::COLOR_BGR2GRAY);
            gray8.convertTo(refL, CV_32F, 1.0/255.0);

            iqa::RegionMasks masks = iqa::computeRegionMasks32(refL);
            cv::Mat vis = visualize_regions(bgr, masks);
            auto parentPath = outPath.parent_path();
            if (!parentPath.empty())
                fs::create_directories(outPath.parent_path());
            if (!cv::imwrite(outPath.string(), vis)) {
                cerr << "Cannot write: " << outPath << endl;
                return 1;
            }
        }
    }
    else if (fs::is_directory(inPath)) {
        // input: catalog
        const fs::path& outDir = outPath;
        fs::create_directories(outDir);
        // collect a list of images to know N
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(inPath)) {
            if (iqa::is_image_file(e.path())) {
                files.push_back(e.path());
            }
        }
        const size_t total = files.size();
        for (size_t i = 0; i < total; ++i) {
            const fs::path& p = files[i];
            // print progress: 5/3000 /path/to/file
            cout << (i+1) << "/" << total << " " << p << endl;
            process_single_file(p, outDir);
        }
    }
    else {
        cerr << "Input is neither file nor directory.\n";
        return 1;
    }
    return 0;
}
