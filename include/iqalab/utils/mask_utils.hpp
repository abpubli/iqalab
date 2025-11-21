#pragma once
#include <opencv2/opencv.hpp>

namespace iqa {

/// Counts pixels where mask(x,y) > threshold.
/// Useful for impulse masks, flat masks, detail masks, etc.
inline std::size_t count_nonzero_threshold(const cv::Mat& mask,
                                           uchar threshold = 0)
{
  CV_Assert(mask.type() == CV_8U);

  std::size_t count = 0;
  const int rows = mask.rows;
  const int cols = mask.cols;

  for (int y = 0; y < rows; ++y) {
    const uchar* row = mask.ptr<uchar>(y);
    for (int x = 0; x < cols; ++x) {
      if (row[x] > threshold)
        ++count;
    }
  }
  return count;
}

} // namespace iqa
