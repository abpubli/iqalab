#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "image_type.hpp"

namespace iqa {

double blocking_score(const cv::Mat& bgr);

double blocking_score_from_file(const std::string& distPath);

cv::Mat flat_blocking_to_mask(const cv::Mat& refBGR, const cv::Mat& distBGR);

} // namespace iqa
