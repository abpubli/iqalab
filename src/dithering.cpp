#include "iqalab/dithering.hpp"

#include "iqalab/color.hpp"
#include "iqalab/impulse.hpp"

#include "iqalab/utils/mask_utils.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

namespace iqa {


std::size_t count_ditherings(const cv::Mat& ditheringMask)
{
    return iqa::count_nonzero_threshold(ditheringMask, 1);
}


// Scan a single row (3 channels) and mark impulsive pixels in rowOut.
//
// cols     – number of pixels in the row
// rowRef   – pointer to CV_32FC3 reference row
// rowDist  – pointer to CV_32FC3 distorted row
// rowOut   – pointer to CV_8U mask row; pixels set to 255 are treated as ditherings.
//
// Heuristic:
//  - for each channel we compute:
//       * sum_diffs: mean |ref - dist| over the row,
//       * sum_dx:    mean |∂x dist| over the row.
//  - a pixel is marked as an dithering if, in that channel, its local gradient
//    is much larger than the gradient in the reference AND
//    its absolute difference to the reference is much larger than the mean
//    difference along the row.
//  - if any channel marks a pixel as dithering, the final mask at that column is 255.
static void detect_ditherings_row_to_mask(int cols, const cv::Vec3f *rowRef, const cv::Vec3f *rowDist, uchar *rowOut) {
    std::memset(rowOut, 0, static_cast<std::size_t>(cols));
    for (int channel = 0; channel < 3; channel++) {
        double sum_diffs = 0;
        for (int bx = 0; bx < cols; bx++) {
            sum_diffs+=fabs(rowRef[bx][channel]- rowDist[bx][channel]);
        }
        double avgDx = 0;
        std::deque<float> min_max_buf;
        double sum_value_acc = 0;
        double sum_dx_acc = 0;
        for (int bx = 0; bx < cols-1; bx++) {
            min_max_buf.push_back(rowDist[bx][channel]);
            if (min_max_buf.size() > 3) min_max_buf.pop_front();
            float mn = *std::min_element(min_max_buf.begin(), min_max_buf.end());
            float mx = *std::max_element(min_max_buf.begin(), min_max_buf.end());
            sum_value_acc += rowDist[bx][channel];
            if (bx >= 8) sum_value_acc -= rowDist[bx-8][channel];
            double avg_win_val = sum_value_acc/std::min(8, bx+1);

            double dxRef = fabs(rowRef[bx+1][channel]-rowRef[bx][channel]);
            double dxDist  = fabs(rowDist[bx+1][channel]-rowDist[bx][channel]);

            float difference = rowDist[bx][channel] - rowRef[bx][channel];
            double mean_diff = avg_win_val - rowRef[bx][channel];
            bool guess_dithering;

            sum_dx_acc += dxDist;
            if (bx >= 8) sum_dx_acc -= fabs(rowDist[bx-7][channel] - rowDist[bx-8][channel]);
            double avg_win_dx = sum_dx_acc/std::min(8, bx+1);

            bool b6 = abs(difference)>=std::max(abs(mean_diff),15.);
            bool b7 = dxDist >= avg_win_dx;
            guess_dithering = b6 && b7;

            if (guess_dithering) {
                rowOut[bx]= 255;
            }
        }
    }
}

// Replace impulsive pixels in a single row using 1D interpolation.
//
// cols      – number of pixels in the row
// rowDist   – distorted input row
// rowMask   – dithering mask row (CV_8U, 0 or non-zero)
// rowOut    – output row
//
// Strategy:
//  - For pixels where mask == 0, we copy rowDist to rowOut.
//  - For leading ditherings (before the first non-dithering), we fill them
//    with the first valid sample in the row.
//  - For ditherings between two valid samples, we linearly interpolate
//    between the left and right valid sample.
//  - For trailing ditherings (after the last valid), we fill them with
//    the last valid sample.
// The function returns the number of dithering pixels in this row.
static int clean_dithering_row(int cols, const cv::Vec3f *rowDist, const uchar* rowMask, cv::Vec3f *rowOut) {
    int lastInserted = -1;
    int dithering_counter = 0;
    for (int bx = 0; bx < cols; bx++) {
        if (rowMask[bx]) {
            dithering_counter++;
        }
        else {
            if (lastInserted < 0) {
                for (int i=0; i<bx; i++)
                    rowOut[i] = rowDist[bx];
            } else {
                int gap = bx - lastInserted;
                if (gap>1) {
                    float w1,w2;
                    for (int i=lastInserted+1; i<bx; i++) {
                        w2 = (i-lastInserted)/float(gap);
                        w1 = 1 - w2;
                        for (int channel=0; channel<3; channel++) {
                            float weighted = w1 * rowDist[lastInserted][channel] + w2 * rowDist[bx][channel];
                            rowOut[i][channel] = weighted;
                        }
                    }
                }
            }
            lastInserted = bx;
            rowOut[bx] = rowDist[bx];
        }
    }
    return dithering_counter;
}

// Detect impulsive artifacts over the entire image.
// For each row, the row-wise detector marks isolated outliers (255)
// while non-dithering pixels remain 0.
cv::Mat dithering_to_mask_lab(const cv::Mat& refLab,
                                 const cv::Mat& distLab)
{
    assert(refLab.size() == distLab.size());
    assert(refLab.type() == CV_32FC3);
    assert(distLab.type() == CV_32FC3);

    cv::Mat ditheringMask(distLab.size(), CV_8U, cv::Scalar(0));

    const int rows = distLab.rows;
    const int cols = distLab.cols;

    for (int y = 0; y < rows; ++y) {
        const auto* rowRef  = refLab.ptr<cv::Vec3f>(y);
        const auto* rowDist = distLab.ptr<cv::Vec3f>(y);
        auto*       rowOut  = ditheringMask.ptr<uchar>(y);
        detect_ditherings_row_to_mask(cols, rowRef, rowDist, rowOut);
    }
    return ditheringMask;
}


struct DualImpulseStats {
    cv::Mat maskLoose;   // mask0
    cv::Mat maskStrict;    // mask1
    std::size_t nImpLoose;
    std::size_t nImpStrict;
    double ratio;
};

static DualImpulseStats compute_dual_dithering_stats_lab(
    const cv::Mat& refLab,
    const cv::Mat& distLAb)
{
    DualImpulseStats s;
    s.maskLoose = dithering_to_mask_lab(refLab, distLAb);
    s.nImpLoose = count_ditherings(s.maskLoose);
    s.ratio = 1;
    return s;
}

cv::Mat dithering_to_mask_bgr8(const cv::Mat& refBGR, const cv::Mat& distBGR,
                     std::size_t& nImp)
{
    assert(refBGR.size() == distBGR.size());
    assert(refBGR.type() == CV_8UC3);
    assert(distBGR.type() == CV_8UC3);

    cv::Mat refBGR32, distBGR32;
    refBGR.convertTo(refBGR32, CV_32FC3);
    distBGR.convertTo(distBGR32, CV_32FC3);

    DualImpulseStats s = compute_dual_dithering_stats_lab(refBGR32, distBGR32);

    if (s.ratio > 7.0) {
        nImp = 0;
        cv::Mat zeroMask(distBGR32.size(), CV_8U, cv::Scalar(0));
        return zeroMask;
    } else {
        nImp = s.nImpLoose;
        return s.maskLoose;
    }
}

ImpulseStats clean_with_mask(const cv::Mat& distBGR32,
                        const cv::Mat& impulseMask,
                        cv::Mat& cleanedBGR32);

// Full pipeline: detection + cleaning.
//
// 1) compute_dithering_mask(ref, dist) to detect impulsive pixels;
// 2) clean_with_mask(dist, mask, cleaned) to inpaint those pixels.
//
// The returned ImpulseStats::count is the total number of pixels modified.
ImpulseStats clean_dithering_lab(const cv::Mat& refLab,
                    const cv::Mat& distLab,
                    cv::Mat& cleanedLab)
{
    DualImpulseStats s = compute_dual_dithering_stats_lab(refLab, distLab);

    cleanedLab.create(distLab.size(), distLab.type());

    if (s.ratio > 7.0) {
        cleanedLab = distLab.clone();
        ImpulseStats stats;
        stats.count = 0;
        return stats;
    } else {
        return clean_with_mask(distLab, s.maskLoose, cleanedLab);
    }
}

// Public BGR8 wrapper: convert to BGR32F, run clean_dithering_bgr32,
// then convert back to BGR8.
ImpulseStats clean_dithering_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR)
{
    assert(refBGR.size() == distBGR.size());
    assert(refBGR.type() == CV_8UC3);
    assert(distBGR.type() == CV_8UC3);

    cv::Mat refLab, distLab;

    bgr8_to_lab32f(refBGR, refLab);
    bgr8_to_lab32f(distBGR,distLab);
    cv::Mat cleanedLab;
    ImpulseStats stats = clean_dithering_lab(refLab, distLab, cleanedLab);
    lab32f_to_bgr8(cleanedLab, outBGR);
    return stats;
}

} // namespace iqa
