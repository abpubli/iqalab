#include <opencv2/core/mat.hpp>

namespace iqa {
cv::Mat blocking_to_mask(const cv::Mat& refRgb, const cv::Mat& distRgb) {
  cv::Mat zeroMask(distRgb.size(), CV_8U, cv::Scalar(0));
  return zeroMask;
}
}