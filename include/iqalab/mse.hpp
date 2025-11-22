#pragma once

#include <opencv2/core.hpp>

namespace iqa::mse
{
// Simple global MSE on a color image.
// We assume the same size and type.
double compute_mse(const cv::Mat& ref, const cv::Mat& test);

// Auxiliary: MSE on a single channel (CV_32F / CV_8U)
double compute_mse_single_channel(const cv::Mat& ref, const cv::Mat& test);

// MSE in Lab for channel (0=L,1=a,2=b), optionally on mask 0/255
double lab_channel_mse(const cv::Mat& labRef,
                       const cv::Mat& labDist,
                       int channel,
                       const cv::Mat& mask = cv::Mat());
}
