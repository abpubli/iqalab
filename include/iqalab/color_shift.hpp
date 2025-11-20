#pragma once
#include <opencv2/core.hpp>

namespace iqa {

struct LabShift {
  double a_L = 1.0;  // slope L
  double b_L = 0.0;  // offset L

  double a_a = 1.0;  // slope a*
  double b_a = 0.0;  // offset a*

  double a_b = 1.0;  // slope b*
  double b_b = 0.0;  // offset b*
};

LabShift compute_lab_shift(const cv::Mat& labRef,
                           const cv::Mat& labDist);

} // namespace iqa
