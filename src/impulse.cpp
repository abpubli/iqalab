#include "iqalab/impulse.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "iqalab/utils/mask_utils.hpp"

namespace iqa {


std::size_t count_impulses(const cv::Mat& impulseMask)
{
    return iqa::count_nonzero_threshold(impulseMask, 0);
}


// Scan a single Lab row (3 channels) and mark impulsive pixels in rowOut.
//
// cols     – number of pixels in the row
// rowRef   – pointer to CV_32FC3 reference row (Lab)
// rowDist  – pointer to CV_32FC3 distorted row (Lab)
// rowOut   – pointer to CV_8U mask row; pixels set to 255 are treated as impulses.
//
// Heuristic:
//  - for each channel we compute:
//       * sum_diffs: mean |ref - dist| over the row,
//       * sum_dx:    mean |∂x dist| over the row.
//  - a pixel is marked as an impulse if, in that channel, its local gradient
//    is much larger than the gradient in the reference AND
//    its absolute difference to the reference is much larger than the mean
//    difference along the row.
//  - if any channel marks a pixel as impulse, the final mask at that column is 255.
static void detect_impulses_row_to_mask(int cols, const cv::Vec3f *rowRef, const cv::Vec3f *rowDist, uchar *rowOut) {
    for (int channel = 0; channel < 3; channel++) {
        double sum_diffs = 0;
        double sum_dx = 0;
        for (int bx = 0; bx < cols; bx++) {
            sum_diffs+=fabs(rowRef[bx][channel]- rowDist[bx][channel]);
        }
        for (int bx = 0; bx < cols-1; bx++) {
            float dx = rowDist[bx+1][channel]-rowDist[bx][channel];
            sum_dx+=std::fabs(dx);
        }
        for (int bx = 0; bx < cols-1; bx++) {
            double dxRef = fabs(rowRef[bx+1][channel]-rowRef[bx][channel]);
            double dxDist  = fabs(rowDist[bx+1][channel]-rowDist[bx][channel]);
            float difference = rowRef[bx][channel] - rowDist[bx][channel];
            bool guess_impulse = dxDist>=2*dxRef &&
                abs(difference)>=2*sum_diffs/cols && dxDist>2*sum_dx/(cols-1);
            if (guess_impulse) {
                rowOut[bx]=255;
            }
        }
    }
}

// Replace impulsive pixels in a single row using 1D interpolation.
//
// cols      – number of pixels in the row
// rowDist   – distorted input row (Lab, CV_32FC3)
// rowMask   – impulse mask row (CV_8U, 0 or non-zero)
// rowOut    – output row (Lab, CV_32FC3)
//
// Strategy:
//  - For pixels where mask == 0, we copy rowDist to rowOut.
//  - For leading impulses (before the first non-impulse), we fill them
//    with the first valid sample in the row.
//  - For impulses between two valid samples, we linearly interpolate
//    between the left and right valid sample.
//  - For trailing impulses (after the last valid), we fill them with
//    the last valid sample.
// The function returns the number of impulse pixels in this row.
static int clean_impulse_row(int cols, const cv::Vec3f *rowDist, const uchar* rowMask, cv::Vec3f *rowOut) {
    int lastInserted = -1;
    int impulse_counter = 0;
    for (int bx = 0; bx < cols; bx++) {
        if (rowMask[bx]) {
            impulse_counter++;
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
    return impulse_counter;
}

// Detect impulsive artifacts over the entire image.
// For each row, the row-wise detector marks isolated Lab outliers (255)
// while non-impulse pixels remain 0.
cv::Mat impulse_to_mask(const cv::Mat& refLab32,
                                 const cv::Mat& distLab32)
{
    assert(refLab32.size() == distLab32.size());
    assert(refLab32.type() == CV_32FC3);
    assert(distLab32.type() == CV_32FC3);

    cv::Mat impulseMask;
    impulseMask.create(distLab32.size(), CV_8U);

    const int rows = distLab32.rows;
    const int cols = distLab32.cols;

    for (int y = 0; y < rows; ++y) {
        const auto* rowRef  = refLab32.ptr<cv::Vec3f>(y);
        const auto* rowDist = distLab32.ptr<cv::Vec3f>(y);
        auto*       rowOut  = impulseMask.ptr<uchar>(y);
        detect_impulses_row_to_mask(cols, rowRef, rowDist, rowOut);
    }
    return impulseMask;
}

/// Clean impulses in Lab space using a precomputed mask.
///
/// distLab32   – CV_32FC3 distorted image in CIE Lab
/// impulseMask – CV_8U, same size, 0/255 mask from compute_impulse_mask()
/// cleanedLab32 – CV_32FC3, output (allocated/created inside)
///
/// The algorithm preserves all pixels where the mask is 0 and replaces
/// masked pixels with values interpolated from neighbouring non-impulse
/// samples along each scan line.
ImpulseStats clean_with_mask(const cv::Mat& distLab32,
                        const cv::Mat& impulseMask,
                        cv::Mat& cleanedLab32)
{
    CV_Assert(distLab32.size() == impulseMask.size());
    CV_Assert(distLab32.type() == CV_32FC3);
    CV_Assert(impulseMask.type() == CV_8U);
    long totalImpulses = 0;
    const int rows = distLab32.rows;
    const int cols = distLab32.cols;
    for (int y = 0; y < rows; ++y) {
        const auto* rowDist = distLab32.ptr<cv::Vec3f>(y);
        const auto*       rowMask  = impulseMask.ptr<uchar>(y);
        auto*             rowOut  = cleanedLab32.ptr<cv::Vec3f>(y);
        totalImpulses += clean_impulse_row(cols, rowDist, rowMask, rowOut);
    }
    ImpulseStats stats;
    stats.count = totalImpulses;
    return stats;
}

// Full Lab pipeline: detection + cleaning.
//
// 1) compute_impulse_mask(ref, dist) to detect impulsive pixels;
// 2) clean_with_mask(dist, mask, cleaned) to inpaint those pixels.
//
// The returned ImpulseStats::count is the total number of pixels modified.
ImpulseStats clean_impulse_lab(const cv::Mat& refLab32,
                                 const cv::Mat& distLab32,
                                 cv::Mat& cleanedLab32) {
    cv::Mat impulseMask = impulse_to_mask(refLab32,distLab32);
    cleanedLab32.create(distLab32.size(), distLab32.type());
    return clean_with_mask(distLab32, impulseMask, cleanedLab32);
}


// Public BGR8 wrapper: convert to Lab32F, run clean_impulse_lab,
// then convert back to BGR8.
ImpulseStats clean_impulse_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR)
{
    assert(refBGR.size() == distBGR.size());
    assert(refBGR.type() == CV_8UC3);
    assert(distBGR.type() == CV_8UC3);

    // Konwersja do 32F BGR [0,1]
    cv::Mat refBGR32, distBGR32;
    refBGR.convertTo(refBGR32, CV_32FC3, 1.0 / 255.0);
    distBGR.convertTo(distBGR32, CV_32FC3, 1.0 / 255.0);

    // Do Lab (32F)
    cv::Mat refLab32, distLab32;
    cv::cvtColor(refBGR32,  refLab32,  cv::COLOR_BGR2Lab);
    cv::cvtColor(distBGR32, distLab32, cv::COLOR_BGR2Lab);

    // Czyszczenie impulsów w przestrzeni Lab
    cv::Mat cleanedLab32;
    ImpulseStats stats = clean_impulse_lab(refLab32, distLab32, cleanedLab32);
    // Powrót do BGR8
    cv::Mat cleanBGR32;
    cv::cvtColor(cleanedLab32, cleanBGR32, cv::COLOR_Lab2BGR);
    cleanBGR32.convertTo(outBGR, CV_8UC3, 255.0);

    return stats;
}

} // namespace iqa
