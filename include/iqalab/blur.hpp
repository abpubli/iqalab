#pragma once

#include <opencv2/core/mat.hpp>

namespace iqa::blur
{

// Relative blur in L channel.
// d = 1 - E_dist / (E_ref + eps), clamped to [0, 1.5].
double relative_blur_L(const cv::Mat& labRef,
                       const cv::Mat& labDist,
                       const cv::Mat& mask = cv::Mat(),
                       double eps = 1e-6);

// Relative blur in a+b (chroma) channels.
double relative_blur_ab(const cv::Mat& labRef,
                        const cv::Mat& labDist,
                        const cv::Mat& mask = cv::Mat(),
                        double eps = 1e-6);

// Relative sharpening / high-frequency increase in L channel.
// Returns max(0, E_dist / (E_ref + eps) - 1), clamped to [0, 1.5].
double relative_sharp_L(const cv::Mat& labRef,
                        const cv::Mat& labDist,
                        const cv::Mat& mask = cv::Mat(),
                        double eps = 1e-6);

// Relative sharpening / high-frequency increase in a+b (chroma) channels.
double relative_sharp_ab(const cv::Mat& labRef,
                         const cv::Mat& labDist,
                         const cv::Mat& mask = cv::Mat(),
                         double eps = 1e-6);

} // namespace iqa::blur
