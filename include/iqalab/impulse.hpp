#pragma once

#include <opencv2/core.hpp>

namespace iqa {

/// Simple statistics returned by impulse removal.
struct ImpulseStats {
    /// Total number of pixels that were classified as impulses
    /// (i.e. pixels where the output differs from the distorted input).
    long count = 0;
};

std::size_t count_impulses(const cv::Mat& impulseMask);

cv::Mat impulse_to_mask_bgr8(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR);

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
