#include "iqalab/iqalab.hpp"

#include <iostream>
#include <opencv2/imgproc.hpp>

namespace iqa {
// Creates a flat region mask based on the Y channel (CV_32F)
static cv::Mat make_flat_mask(const cv::Mat& y32f, const float laplacianThresh)
{
    cv::Mat lap;
    cv::Laplacian(y32f, lap, CV_32F, 3);

    cv::Mat absLap;
    cv::absdiff(lap, 0, absLap);

    cv::Mat flatMask(y32f.size(), CV_8U);
    for (int y = 0; y < y32f.rows; ++y) {
        const float* row = absLap.ptr<float>(y);
        auto* mrow = flatMask.ptr<uchar>(y);
        for (int x = 0; x < y32f.cols; ++x) {
            // jeśli gradient niski -> piksel "płaski"
            mrow[x] = (row[x] < laplacianThresh) ? 255 : 0;
        }
    }
    return flatMask;
}

// Calculates the “blocking score” for a single channel (CV_32F, 1 channel)
// If flatMask != nullptr, only uses pixels where flatMask(y,x) != 0.
double blocking_score_channel(const cv::Mat& ch32f,
                                                int blockSize,
                                                const cv::Mat* flatMask = nullptr)
{
    const int w = ch32f.cols;
    const int h = ch32f.rows;

    if (w < blockSize * 2 || h < blockSize * 2)
        return 1.0; // image too small, treated as no blocking

    // horizontal differences (between columns)
    cv::Mat diffX(h, w - 1, CV_32F);
    for (int y = 0; y < h; ++y) {
        const float* row = ch32f.ptr<float>(y);
        float* drow = diffX.ptr<float>(y);
        for (int x = 0; x < w - 1; ++x) {
            drow[x] = std::fabs(row[x+1] - row[x]);
        }
    }

    // vertical differences (between rows)
    cv::Mat diffY(h - 1, w, CV_32F);
    for (int y = 0; y < h - 1; ++y) {
        const float* row1 = ch32f.ptr<float>(y);
        const float* row2 = ch32f.ptr<float>(y+1);
        float* drow = diffY.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            drow[x] = std::fabs(row2[x] - row1[x]);
        }
    }

    auto isFlat = [&](int y, int x) -> bool {
        if (!flatMask) return true;
        return flatMask->at<uchar>(y, x) != 0;
    };

    // --- VERTICAL block boundaries (x = 8,16,24,...) ---
    double sumBoundaryX = 0.0;
    int countBoundaryX = 0;

    for (int x = blockSize; x < w - 1; x += blockSize) {
        for (int y = 0; y < h; ++y) {
            if (!isFlat(y, x-1)) continue;
            sumBoundaryX += diffX.at<float>(y, x - 1);
            ++countBoundaryX;
        }
    }

    double sumInnerX = 0.0;
    int countInnerX = 0;

    for (int x = 1; x < w - 1; ++x) {
        if (x % blockSize == 0) continue; // we ignore block boundaries
        for (int y = 0; y < h; ++y) {
            if (!isFlat(y, x)) continue;
            sumInnerX += diffX.at<float>(y, x);
            ++countInnerX;
        }
    }

    double meanBoundaryX = (countBoundaryX > 0) ? (sumBoundaryX / countBoundaryX) : 0.0;
    double meanInnerX    = (countInnerX   > 0) ? (sumInnerX   / countInnerX)   : 1e-6;

    // --- HORIZONTAL block boundaries (y = 8,16,24,...) ---
    double sumBoundaryY = 0.0;
    int countBoundaryY = 0;
    for (int y = blockSize; y < h - 1; y += blockSize) {
        for (int x = 0; x < w; ++x) {
            if (!isFlat(y-1, x)) continue;
            sumBoundaryY += diffY.at<float>(y - 1, x);
            ++countBoundaryY;
        }
    }

    double sumInnerY = 0.0;
    int countInnerY = 0;
    for (int y = 1; y < h - 1; ++y) {
        if (y % blockSize == 0) continue;
        for (int x = 0; x < w; ++x) {
            if (!isFlat(y, x)) continue;
            sumInnerY += diffY.at<float>(y, x);
            ++countInnerY;
        }
    }

    double meanBoundaryY = (countBoundaryY > 0) ? (sumBoundaryY / countBoundaryY) : 0.0;
    double meanInnerY    = (countInnerY   > 0) ? (sumInnerY   / countInnerY)   : 1e-6;

    double scoreX = meanBoundaryX / meanInnerX;
    double scoreY = meanBoundaryY / meanInnerY;

    // average of vertical and horizontal
    double score = 0.5 * (scoreX + scoreY);
    return score;
}

double blocking_score(const cv::Mat& bgr)
{
    CV_Assert(!bgr.empty());
    CV_Assert(bgr.channels() == 3);

    cv::Mat ycrcb;
    cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);

    // Podziel kanały
    std::vector<cv::Mat> ch;
    cv::split(ycrcb, ch);
    // ch[0] = Y, ch[1] = Cr, ch[2] = Cb

    // Convert to float 0..1
    cv::Mat y32f, cr32f, cb32f;
    ch[0].convertTo(y32f,  CV_32F, 1.0/255.0);
    ch[1].convertTo(cr32f, CV_32F, 1.0/255.0);
    ch[2].convertTo(cb32f, CV_32F, 1.0/255.0);

    // -------- RANGE DETECTION (max-min) --------
    double range[3];
    for (int i = 0; i < 3; i++) {
        double mn, mx;
        cv::minMaxLoc(ch[i], &mn, &mx);
        range[i] = mx - mn;  // w skali 0..255
    }

    // We assume:
    //  - Y always has significant weight
    //  - Cr/Cb is ignored if range < 1.5 (almost constant)
    auto weight_from_range = [](double r) -> double {
        if (r < 1.5) return 0.0;                 // skip channel
        double w = r / 20.0;                    // heuristics
        if (w > 1.0) w = 1.0;
        return w;
    };

    double wY  = 1.0;                       // luminance always important
    double wCr = weight_from_range(range[1]);
    double wCb = weight_from_range(range[2]);

    // If all channels are flat → return 0
    if (wY == 0.0 && wCr == 0.0 && wCb == 0.0)
        return 0.0;

    const int blockSize = 8;

    // mask of flat regions Y
    cv::Mat flatMask = make_flat_mask(y32f, 2.0f);

    double scoreY  = blocking_score_channel(y32f,  blockSize, &flatMask);
    double scoreCr = (wCr > 0.0 ? blocking_score_channel(cr32f, blockSize, &flatMask) : 0.0);
    double scoreCb = (wCb > 0.0 ? blocking_score_channel(cb32f, blockSize, &flatMask) : 0.0);

    // we mix the weighted average with absolutely minimal changes
    double weighted =
        (wY  * scoreY  +
         wCr * scoreCr +
         wCb * scoreCb) / (wY + wCr + wCb + 1e-12);

    // we cut off baseline = 1.0
    double blocking = weighted - 1.0;
    if (blocking < 0.0) blocking = 0.0;

    return blocking;
}
}
