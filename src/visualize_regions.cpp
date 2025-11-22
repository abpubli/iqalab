#include "iqalab/region_masks.hpp"

#include <iqalab/visualize_regions.hpp>
#include <opencv2/core/mat.hpp>

namespace iqa {
struct RegionMasks;

cv::Mat visualize_regions(const cv::Mat& bgr, const RegionMasks& masks)
{
  cv::Mat vis = bgr.clone();

  // kolorowanie flat/mid/detail
  // przykład:

  for (int y = 0; y < vis.rows; ++y) {
    const uchar* f = masks.flat.ptr<uchar>(y);
    const uchar* m = masks.mid.ptr<uchar>(y);
    const uchar* d = masks.detail.ptr<uchar>(y);

    cv::Vec3b* outRow = vis.ptr<cv::Vec3b>(y);

    for (int x = 0; x < vis.cols; ++x) {
      if (f[x] == 255) {
        outRow[x] = cv::Vec3b(255, 0, 0);       // blue for flat
      } else if (m[x] == 255) {
        outRow[x] = cv::Vec3b(0, 255, 255);     // yellow for mid
      } else if (d[x] == 255) {
        outRow[x] = cv::Vec3b(0, 0, 255);       // red for detail
      }
    }
  }

  return vis;
}

} // namespace iqa
