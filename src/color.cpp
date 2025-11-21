#include "iqalab/color.hpp"

#include "iqalab/debug.hpp"

#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace iqa {
// BGR8 (CV_8UC3, 0..255) -> Lab32 (CV_32FC3, L in [0..100])
void bgr8_to_lab32f(const cv::Mat& bgr8, cv::Mat& lab32f)
{
  CV_Assert(bgr8.type() == CV_8UC3);
  cv::Mat bgr32f;
  bgr8.convertTo(bgr32f, CV_32F, 1.0 / 255.0);
  cv::cvtColor(bgr32f, lab32f, cv::COLOR_BGR2Lab);
  debug_assert_normalized_01(lab32f,"lab32f");
}

void bgr32_to_lab32f(const cv::Mat& bgr32, cv::Mat& lab32f)
{
  CV_Assert(bgr32.type() == CV_32FC3);
  cv::Mat bgrNorm32f;
  debug_assert_normalized_0255(bgr32,"bgr32");
  bgr32.convertTo(bgrNorm32f, CV_32F, 1.0 / 255.0);
  cv::cvtColor(bgr32, lab32f, cv::COLOR_BGR2Lab);
}

void bgr32norm_to_lab32f(const cv::Mat& bgr32, cv::Mat& lab32f)
{
  CV_Assert(bgr32.type() == CV_32FC3);
  cv::Mat bgrNorm32f;
  debug_assert_normalized_01(bgr32,"bgr32");
  cv::cvtColor(bgr32, lab32f, cv::COLOR_BGR2Lab);
}
}