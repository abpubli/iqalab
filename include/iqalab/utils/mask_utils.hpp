#pragma once
#include <opencv2/opencv.hpp>

namespace iqa {

/// Counts pixels where mask(x,y) > threshold.
/// Useful for impulse masks, flat masks, detail masks, etc.
std::size_t count_nonzero_threshold(const cv::Mat& mask,
                                           uchar threshold = 1);

// Build 8-bit single-channel diff mask:
// diff(y,x) = max( |B1-B2|, |G1-G2|, |R1-R2| )
cv::Mat make_channel_max_diff_mask(const cv::Mat& img1_bgr,
                                   const cv::Mat& img2_bgr);

} // namespace iqa
