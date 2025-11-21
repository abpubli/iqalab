#pragma once
#include <opencv2/core.hpp>

namespace iqa {

struct ImpulseStats {
  int count = 0;  // total impulses removed
};


ImpulseStats clean_impulse_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR);

// High-level API working in Lab (CV_32FC3, range as in your existing code:
// L in ~[0,100], a*, b* ~[-128,128]).
//
// refLab32  – reference image in Lab, CV_32FC3
// distLab32 – distorted image in Lab, CV_32FC3
// cleanedLab32 – output (same type/size as distLab32)
//
// Returns ImpulseStats with the number of impulses detected/cleaned.
ImpulseStats clean_impulse_lab(const cv::Mat& refLab32,
                                 const cv::Mat& distLab32,
                                 cv::Mat& cleanedLab32);

// Convenience wrapper for BGR8 images (typowy case dla plików z dysku).
//
// refBGR  – CV_8UC3
// distBGR – CV_8UC3
// outBGR  – CV_8UC3 (cleaned)
ImpulseStats clean_impulse_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR);

} // namespace iqa
