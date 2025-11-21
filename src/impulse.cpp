#include "iqalab/impulse.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "iqalab/utils/mask_utils.hpp"

namespace iqa {


std::size_t count_impulses(const cv::Mat& impulseMask)
{
    return iqa::count_nonzero_threshold(impulseMask, 1);
}


// Scan a single row (3 channels) and mark impulsive pixels in rowOut.
//
// cols     – number of pixels in the row
// rowRef   – pointer to CV_32FC3 reference row
// rowDist  – pointer to CV_32FC3 distorted row
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
static void detect_impulses_row_to_mask(int cols, const cv::Vec3f *rowRef, const cv::Vec3f *rowDist, uchar *rowOut, bool strict) {
    std::memset(rowOut, 0, static_cast<std::size_t>(cols));
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

            sum_dx_acc += dxDist;
            if (bx >= 8) sum_dx_acc -= fabs(rowDist[bx-7][channel] - rowDist[bx-8][channel]);
            double avg_win_dx = sum_dx_acc/std::min(8, bx+1);

            float difference = rowDist[bx][channel] - rowRef[bx][channel];
            double avgDx = sum_dx/(cols-1);
            double mean_diff = avg_win_val - rowRef[bx][channel];
            //strict
            bool b0 = rowDist[bx][channel]>=100|| rowDist[bx][channel]<=26;
            bool b1 = (rowDist[bx][channel] == mx || rowDist[bx][channel] == mn);
            bool b2 = abs(difference)>=40;
            bool b3 = dxDist>=2*dxRef;
            bool b4 = dxDist>4*avgDx;
            bool b5 = abs(difference)>=std::max(abs(mean_diff),15.);
            //loose
            bool b6 = abs(difference)>=std::max(abs(mean_diff),15.);
            bool b7 = dxDist >= avg_win_dx;
            bool guess_impulse_loose = b6 && b7;

            bool guess_impulse_strict = b0 && b1 && b2 && b3 && b4 && b5;
            bool guess_impulse = strict?guess_impulse_strict:guess_impulse_loose;
            if (guess_impulse) {
                rowOut[bx]= 255;
            }
        }
    }
}

// Replace impulsive pixels in a single row using 1D interpolation.
//
// cols      – number of pixels in the row
// rowDist   – distorted input row
// rowMask   – impulse mask row (CV_8U, 0 or non-zero)
// rowOut    – output row
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

void detect_impulses_row_to_mask(int cols, const cv::Vec<float, 3> * vec,
                                unsigned char * row_out, bool strict);
// Detect impulsive artifacts over the entire image.
// For each row, the row-wise detector marks isolated outliers (255)
// while non-impulse pixels remain 0.
cv::Mat impulse_to_mask_bgr32(const cv::Mat& refBGR32,
                                 const cv::Mat& distBGR32, bool strict)
{
    assert(refBGR32.size() == distBGR32.size());
    assert(refBGR32.type() == CV_32FC3);
    assert(distBGR32.type() == CV_32FC3);

    cv::Mat impulseMask(distBGR32.size(), CV_8U, cv::Scalar(0));

    const int rows = distBGR32.rows;
    const int cols = distBGR32.cols;

    for (int y = 0; y < rows; ++y) {
        const auto* rowRef  = refBGR32.ptr<cv::Vec3f>(y);
        const auto* rowDist = distBGR32.ptr<cv::Vec3f>(y);
        auto*       rowOut  = impulseMask.ptr<uchar>(y);
        detect_impulses_row_to_mask(cols, rowRef, rowDist, rowOut, strict);
    }
    return impulseMask;
}


struct DualImpulseStats {
    cv::Mat maskStrict;   // mask0
    cv::Mat maskLoose;    // mask1
    std::size_t nImpStrict;
    std::size_t nImpLoose;
    double ratio;         // (nImpStrict+1)/(nImpLoose+0.1)
};

static DualImpulseStats compute_dual_impulse_stats_bgr32(
    const cv::Mat& refBGR32,
    const cv::Mat& distBGR32)
{
    DualImpulseStats s;

    s.maskStrict = impulse_to_mask_bgr32(refBGR32, distBGR32, false);
    s.maskLoose  = impulse_to_mask_bgr32(refBGR32, distBGR32, true);

    s.nImpStrict = count_impulses(s.maskStrict);
    s.nImpLoose  = count_impulses(s.maskLoose);

    s.ratio = (static_cast<double>(s.nImpStrict) + 0.1) /
              (static_cast<double>(s.nImpLoose)  + 0.1);

    return s;
}

cv::Mat impulse_to_mask_bgr8(const cv::Mat& refBGR, const cv::Mat& distBGR,
                     std::size_t& nImp)
{
    assert(refBGR.size() == distBGR.size());
    assert(refBGR.type() == CV_8UC3);
    assert(distBGR.type() == CV_8UC3);

    cv::Mat refBGR32, distBGR32;
    refBGR.convertTo(refBGR32, CV_32FC3);
    distBGR.convertTo(distBGR32, CV_32FC3);

    DualImpulseStats s = compute_dual_impulse_stats_bgr32(refBGR32, distBGR32);

    if (s.ratio > 7.0) {
        nImp = 0;
        cv::Mat zeroMask(distBGR32.size(), CV_8U, cv::Scalar(0));
        return zeroMask;
    } else {
        nImp = s.nImpStrict;
        return s.maskStrict;
    }
}

/// Clean impulses using a precomputed mask.
///
/// distBGR32   – CV_32FC3 distorted image
/// impulseMask – CV_8U, same size, 0/255 mask from compute_impulse_mask()
/// cleanedBGR32 – CV_32FC3, output (allocated/created inside)
///
/// The algorithm preserves all pixels where the mask is 0 and replaces
/// masked pixels with values interpolated from neighbouring non-impulse
/// samples along each scan line.
ImpulseStats clean_with_mask(const cv::Mat& distBGR32,
                        const cv::Mat& impulseMask,
                        cv::Mat& cleanedBGR32)
{
    CV_Assert(distBGR32.size() == impulseMask.size());
    CV_Assert(distBGR32.type() == CV_32FC3);
    CV_Assert(impulseMask.type() == CV_8U);
    size_t totalImpulses = 0;
    const int rows = distBGR32.rows;
    const int cols = distBGR32.cols;
    for (int y = 0; y < rows; ++y) {
        const auto* rowDist = distBGR32.ptr<cv::Vec3f>(y);
        const auto*       rowMask  = impulseMask.ptr<uchar>(y);
        auto*             rowOut  = cleanedBGR32.ptr<cv::Vec3f>(y);
        totalImpulses += clean_impulse_row(cols, rowDist, rowMask, rowOut);
    }
    ImpulseStats stats;
    stats.count = totalImpulses;
    return stats;
}

// Full pipeline: detection + cleaning.
//
// 1) compute_impulse_mask(ref, dist) to detect impulsive pixels;
// 2) clean_with_mask(dist, mask, cleaned) to inpaint those pixels.
//
// The returned ImpulseStats::count is the total number of pixels modified.
ImpulseStats clean_impulse_bgr32(const cv::Mat& refBGR32,
                    const cv::Mat& distBGR32,
                    cv::Mat& cleanedBGR32)
{
    DualImpulseStats s = compute_dual_impulse_stats_bgr32(refBGR32, distBGR32);

    cleanedBGR32.create(distBGR32.size(), distBGR32.type());

    // zachowujesz dotychczasowe logowanie (np. z '|')
    std::cout << s.ratio << "|";

    if (s.ratio > 7.0) {
        cleanedBGR32 = distBGR32.clone();
        ImpulseStats stats;
        stats.count = 0;
        return stats;
    } else {
        return clean_with_mask(distBGR32, s.maskStrict, cleanedBGR32);
    }
}

// Public BGR8 wrapper: convert to BGR32F, run clean_impulse_bgr32,
// then convert back to BGR8.
ImpulseStats clean_impulse_image(const cv::Mat& refBGR,
                                 const cv::Mat& distBGR,
                                 cv::Mat& outBGR)
{
    assert(refBGR.size() == distBGR.size());
    assert(refBGR.type() == CV_8UC3);
    assert(distBGR.type() == CV_8UC3);

    cv::Mat refBGR32, distBGR32;
    refBGR.convertTo(refBGR32, CV_32FC3);
    distBGR.convertTo(distBGR32, CV_32FC3);

    cv::Mat cleanedBGR32;
    ImpulseStats stats = clean_impulse_bgr32(refBGR32, distBGR32, cleanedBGR32);
    cleanedBGR32.convertTo(outBGR, CV_8UC3);

    return stats;
}

} // namespace iqa
