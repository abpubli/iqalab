#pragma once

#include <opencv2/core.hpp>

namespace iqa::blur
{
// L gradient energy (sharpness) – Lab, CV_32FC3, optional mask 0/255
double l_channel_gradient_energy(const cv::Mat& lab,
                                 const cv::Mat& mask = cv::Mat());

// Relative blur (full-reference) in the L channel:
// 0.0  -> no additional blur
// 1.0+ -> strong additional blur
double relative_blur_L(const cv::Mat& labRef,
                       const cv::Mat& labDist,
                       const cv::Mat& mask = cv::Mat(),
                       double eps = 1e-6);

double relative_blur_ab(const cv::Mat& labRef,
                        const cv::Mat& labDist,
                        const cv::Mat& mask = cv::Mat(),
                        double eps = 1e-6);

// (optional later) multi-scale blur, etc.
}
