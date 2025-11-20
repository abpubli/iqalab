#include "iqalab/iqalab.hpp"

#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
namespace iqa {

double blocking_score_from_file(const std::string& distPath)
{
    cv::Mat distBGR = cv::imread(distPath, cv::IMREAD_COLOR);
    if (distBGR.empty()) {
        throw std::runtime_error("blocking_score_from_file: failed to read image: " + distPath);
    }
    return blocking_score(distBGR);
}
} // namespace iqa
