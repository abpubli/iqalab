#pragma once

#include <opencv2/core.hpp>

namespace iqa {

/// Simple statistics returned by impulse removal.
struct ImpulseStats {
    /// Total number of pixels that were classified as impulses
    /// (i.e. pixels where the output differs from the distorted input).
    long count = 0;
};


cv::Mat impulse_to_mask(const cv::Mat& refLab32,
                                 const cv::Mat& distLab32);

ImpulseStats clean_impulse_lab(const cv::Mat& refLab32,
                               const cv::Mat& distLab32,
                               cv::Mat& cleanedLab32);

/// Convenience wrapper for 8-bit BGR images.
///
/// refBGR  – CV_8UC3, reference image in BGR
/// distBGR – CV_8UC3, distorted image in BGR
/// outBGR  – CV_8UC3, cleaned output; allocated/overwritten inside.
///
/// Internally converts both images to Lab32F, runs clean_impulse_lab(),
/// and converts the result back to BGR8.
ImpulseStats clean_impulse_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR);

} // namespace iqa
