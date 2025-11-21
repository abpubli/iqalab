#pragma once

namespace cv {
class Mat;
}

static inline void debug_assert_normalized_01(const cv::Mat& m,const char* name)
{
#ifdef _DEBUG
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(m, &minv, &maxv);

  // wypis
  fprintf(stderr, "[DEBUG] %s: min=%f max=%f\n", name, minv, maxv);

  // asercje
  DBG_ASSERT_MSG(maxv <= 1.0 + 1e-6,
      "Matrix expected to be in 0..1 range");
#endif
}

static inline void debug_assert_normalized_0255(const cv::Mat& m,const char* name)
{
#ifdef _DEBUG
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(m, &minv, &maxv);

  // wypis
  fprintf(stderr, "[DEBUG] %s: min=%f max=%f\n", name, minv, maxv);

  // asercje
  DBG_ASSERT_MSG(maxv >= 1.5 && maxv <= 255 + 1e-6,
      "Matrix expected to be in 0..1 range");
#endif
}
