#pragma once
#include <opencv2/core.hpp>
namespace iqa {

struct RegionMasks {
  cv::Mat flatMask;    // CV_8U, 255 = flat
  cv::Mat detailMask;  // CV_8U, 255 = detail
  cv::Mat midMask;     // CV_8U, 255 = pośrednie
  cv::Mat gradMag;     // CV_32F, |grad| refL
};

// refL: CV_32F, channel L* or gray
RegionMasks computeRegionMasks32(const cv::Mat& refL,
                               float flatPercentile   = 0.3f,
                               float detailPercentile = 0.7f);

struct ImpulseScore {
  double meanOnFlat;   // average |ref - dist| on flat
  double p95OnFlat;    // 95th percentile |ref - dist| on flat
  int    countFlat;    // number of flat samples on which the count was performed
};

ImpulseScore score_impulses(const cv::Mat& refL,
                            const cv::Mat& distL,
                            const RegionMasks& masks);

struct BlurScore {
  double meanLossOnDetail;  // average gradient loss per detail (magRef - magDist, >0)
  double p95LossOnDetail;   // 95th percentile of gradient loss
  int    countDetail;       // number of samples detail
};

BlurScore score_blur(const cv::Mat& refL,
                     const cv::Mat& distL,
                     const RegionMasks& masks);

} // namespace iqa
