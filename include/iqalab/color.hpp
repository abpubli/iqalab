#pragma once
#include <opencv2/core/mat.hpp>

namespace iqa {
  void bgr8_to_lab32f(const cv::Mat &bgr8, cv::Mat &lab32f);
  void bgr32_to_lab32f(const cv::Mat& bgr32, cv::Mat& lab32f);
  void bgr32norm_to_lab32f(const cv::Mat& bgr32, cv::Mat& lab32f);
} // namespace iqa