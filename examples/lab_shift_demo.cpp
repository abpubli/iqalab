#include "iqalab/color_shift.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

#include "iqalab/iqalab.hpp"

using namespace iqa;

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cout << "Usage: lab_shift_demo <ref_image> <dist_image>\n";
    return 1;
  }

  std::string refPath  = argv[1];
  std::string distPath = argv[2];

  cv::Mat refBGR  = cv::imread(refPath, cv::IMREAD_COLOR);
  cv::Mat distBGR = cv::imread(distPath, cv::IMREAD_COLOR);

  if (refBGR.empty() || distBGR.empty()) {
    std::cerr << "Failed to load images.\n";
    return 1;
  }

  // Convert to Lab (float)
  cv::Mat refLab32, distLab32;

  cv::Mat refLab, distLab;
  cv::cvtColor(refBGR,  refLab,  cv::COLOR_BGR2Lab);
  cv::cvtColor(distBGR, distLab, cv::COLOR_BGR2Lab);

  refLab.convertTo(refLab32,   CV_32FC3);
  distLab.convertTo(distLab32, CV_32FC3);

  // Compute shift model
  LabShift shift = compute_lab_shift(refLab32, distLab32);

  std::cout << "Computed global Lab linear shift:\n";
  std::cout << " L*: a=" << shift.a_L << "   b=" << shift.b_L << "\n";
  std::cout << " a*: a=" << shift.a_a << "   b=" << shift.b_a << "\n";
  std::cout << " b*: a=" << shift.a_b << "   b=" << shift.b_b << "\n";

  return 0;
}
