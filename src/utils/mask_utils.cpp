#include "iqalab/utils/mask_utils.hpp"
std::size_t iqa::count_nonzero_threshold(const cv::Mat &mask, uchar threshold) {
  CV_Assert(mask.type() == CV_8U);

  std::size_t count = 0;
  const int rows = mask.rows;
  const int cols = mask.cols;

  for (int y = 0; y < rows; ++y) {
    const uchar *row = mask.ptr<uchar>(y);
    for (int x = 0; x < cols; ++x) {
      if (row[x] >= threshold)
        ++count;
    }
  }
  return count;
}

cv::Mat iqa::make_channel_max_diff_mask(const cv::Mat &img1_bgr,
                                        const cv::Mat &img2_bgr) {
  if (img1_bgr.size() != img2_bgr.size() ||
      img1_bgr.type() != img2_bgr.type() || img1_bgr.type() != CV_8UC3) {
    throw std::runtime_error("make_channel_max_diff_mask: both images must be "
                             "CV_8UC3 and same size");
  }

  cv::Mat diff(img1_bgr.size(), CV_8UC1);

  const int rows = img1_bgr.rows;
  const int cols = img1_bgr.cols;

  for (int y = 0; y < rows; ++y) {
    const cv::Vec3b *row1 = img1_bgr.ptr<cv::Vec3b>(y);
    const cv::Vec3b *row2 = img2_bgr.ptr<cv::Vec3b>(y);
    uchar *drow = diff.ptr<uchar>(y);

    for (int x = 0; x < cols; ++x) {
      const cv::Vec3b &p1 = row1[x];
      const cv::Vec3b &p2 = row2[x];

      int db = std::abs(int(p1[0]) - int(p2[0]));
      int dg = std::abs(int(p1[1]) - int(p2[1]));
      int dr = std::abs(int(p1[2]) - int(p2[2]));

      int m = std::max({db, dg, dr});
      drow[x] = static_cast<uchar>(m);
    }
  }

  return diff;
}