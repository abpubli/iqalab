#pragma once

#include <opencv2/core.hpp>

namespace iqa
{
// Simple global MSE on a color image.
// We assume the same size and type.
double compute_mse(const cv::Mat& ref, const cv::Mat& test);

// Auxiliary: MSE on a single channel (CV_32F / CV_8U)
double compute_mse_single_channel(const cv::Mat& ref, const cv::Mat& test);
}
