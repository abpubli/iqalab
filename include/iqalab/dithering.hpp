#pragma once

#include "impulse.hpp"

#include <opencv2/core.hpp>

namespace iqa {

std::size_t count_ditherings(const cv::Mat& ditheringMask);

cv::Mat dithering_to_mask_bgr8(const cv::Mat& refBGR, const cv::Mat& distBGR,
                        std::size_t& nImp);

/// Convenience wrapper for 8-bit BGR images.
///
/// refBGR  – CV_8UC3, reference image in BGR
/// distBGR – CV_8UC3, distorted image in BGR
/// outBGR  – CV_8UC3, cleaned output; allocated/overwritten inside.
///
/// Internally converts both images to Lab32F, runs clean_dithering_lab(),
/// and converts the result back to BGR8.
ImpulseStats clean_dithering_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR);

} // namespace iqa
