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

// Lab32f (CV_32FC3, L in [0..100]) -> BGR8 (CV_8UC3, 0..255)
void lab32f_to_bgr8(const cv::Mat& lab32f, cv::Mat& bgr8)
{
  CV_Assert(lab32f.type() == CV_32FC3);

#ifdef _DEBUG
  // Simple check of Lab channel ranges – helps catch silly mistakes
  std::vector<cv::Mat> ch;
  cv::split(lab32f, ch);
  CV_Assert(ch.size() == 3);

  double minL, maxL, mina, maxa, minb, maxb;
  cv::minMaxLoc(ch[0], &minL, &maxL);
  cv::minMaxLoc(ch[1], &mina, &maxa);
  cv::minMaxLoc(ch[2], &minb, &maxb);

  // L should be ~[0..100], a/b ~[-128..127] (in practice, a little tighter)
  // We don't do hard asserts on the range, just sanity checks
  if (std::isnan(minL) || std::isnan(maxL) ||
      std::isnan(mina) || std::isnan(maxa) ||
      std::isnan(minb) || std::isnan(maxb)) {
    CV_Assert(!"lab32f_to_bgr8: Lab contains NaN");
      }

  CV_Assert(minL >= -1.0 && maxL <= 110.0);
  CV_Assert(mina >= -200.0 && maxa <= 200.0);
  CV_Assert(minb >= -200.0 && maxb <= 200.0);
#endif

  // 1) Lab32f -> BGR32f (0..1)
  cv::Mat bgr32f;
  cv::cvtColor(lab32f, bgr32f, cv::COLOR_Lab2BGR);

#ifdef _DEBUG
  // Checking whether BGR is sensibly normalized (0..1)
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(bgr32f, &minv, &maxv);
  if (std::isnan(minv) || std::isnan(maxv)) {
    CV_Assert(!"lab32f_to_bgr8: BGR32f contains NaN after COLOR_Lab2BGR");
  }
  // In practice, there may be minimal overshoots of the type [-1e-3, 1+1e-3].
  CV_Assert(minv > -1.0 && maxv < 2.0);
#endif

  // 2) BGR32f (0..1) -> BGR8 (0..255, saturate cast)
  bgr32f.convertTo(bgr8, CV_8UC3, 255.0);
}

}