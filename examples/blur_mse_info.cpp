#include <iqalab/blur.hpp>
#include <iqalab/mse.hpp>
#include <iqalab/region_masks.hpp>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Pair { std::string refPath, distPath; };

static bool load_image_lab32(const std::string& path, cv::Mat& lab)
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

static std::vector<Pair> load_pairs(const std::string& arg1, int argc)
{
    std::vector<Pair> pairs;
    if (argc == 3)
    {
        pairs.push_back(Pair{arg1, std::string()}); // nieużywane, zaraz nadpiszemy
    }
    return pairs;
}

static std::vector<Pair> load_pairs_file(const std::string& path)
{
    std::vector<Pair> pairs;
    std::ifstream in(path);
    std::string ref, dist;
    while (in >> ref >> dist)
        pairs.push_back({ref, dist});
    return pairs;
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 2)
    {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " ref.png dist.png\n";
        std::cerr << "  " << argv[0] << " pairs.txt\n";
        return 1;
    }

    std::vector<Pair> pairs;
    if (argc == 3)
    {
        pairs.push_back(Pair{argv[1], argv[2]});
    }
    else
    {
        pairs = load_pairs_file(argv[1]);
        if (pairs.empty())
        {
            std::cerr << "No pairs.\n";
            return 1;
        }
    }

    std::cout
        << "ref_path,dist_path,"
        << "blur_L_global,"
        << "blur_L_flat,blur_L_mid,blur_L_detail,"
        << "mse_L_all,mse_a_all,mse_b_all,"
        << "mse_L_flat,mse_L_mid,mse_L_detail,"
        << "mse_a_flat,mse_a_mid,mse_a_detail,"
        << "mse_b_flat,mse_b_mid,mse_b_detail"
        << "\n";

    for (const auto& p : pairs)
    {
        cv::Mat labRef, labDist;
        if (!load_image_lab32(p.refPath, labRef) ||
            !load_image_lab32(p.distPath, labDist))
            continue;

        if (labRef.size() != labDist.size())
        {
            std::cerr << "Size mismatch: " << p.refPath << " vs " << p.distPath << "\n";
            continue;
        }

        iqa::RegionMasks masks = iqa::compute_region_masks(labRef);

        double blur_L_global = iqa::blur::relative_blur_L(labRef, labDist);
        double blur_L_flat   = iqa::blur::relative_blur_L(labRef, labDist, masks.flat);
        double blur_L_mid    = iqa::blur::relative_blur_L(labRef, labDist, masks.mid);
        double blur_L_detail = iqa::blur::relative_blur_L(labRef, labDist, masks.detail);

        double mse_L_all = iqa::mse::lab_channel_mse(labRef, labDist, 0);
        double mse_a_all = iqa::mse::lab_channel_mse(labRef, labDist, 1);
        double mse_b_all = iqa::mse::lab_channel_mse(labRef, labDist, 2);

        double mse_L_flat   = iqa::mse::lab_channel_mse(labRef, labDist, 0, masks.flat);
        double mse_L_mid    = iqa::mse::lab_channel_mse(labRef, labDist, 0, masks.mid);
        double mse_L_detail = iqa::mse::lab_channel_mse(labRef, labDist, 0, masks.detail);

        double mse_a_flat   = iqa::mse::lab_channel_mse(labRef, labDist, 1, masks.flat);
        double mse_a_mid    = iqa::mse::lab_channel_mse(labRef, labDist, 1, masks.mid);
        double mse_a_detail = iqa::mse::lab_channel_mse(labRef, labDist, 1, masks.detail);

        double mse_b_flat   = iqa::mse::lab_channel_mse(labRef, labDist, 2, masks.flat);
        double mse_b_mid    = iqa::mse::lab_channel_mse(labRef, labDist, 2, masks.mid);
        double mse_b_detail = iqa::mse::lab_channel_mse(labRef, labDist, 2, masks.detail);

        std::cout
            << p.refPath << "," << p.distPath << ","
            << blur_L_global << ","
            << blur_L_flat << "," << blur_L_mid << "," << blur_L_detail << ","
            << mse_L_all << "," << mse_a_all << "," << mse_b_all << ","
            << mse_L_flat << "," << mse_L_mid << "," << mse_L_detail << ","
            << mse_a_flat << "," << mse_a_mid << "," << mse_a_detail << ","
            << mse_b_flat << "," << mse_b_mid << "," << mse_b_detail
            << "\n";
    }

    return 0;
}
