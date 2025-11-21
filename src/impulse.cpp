#include "iqalab/impulse.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace iqa {

// Returns the pulse counter in a single line.
int clean_impulse_row(int cols, const cv::Vec3f *rowRef, const cv::Vec3f *rowDist, cv::Vec3f *rowOut) {
    std::vector<bool> impulse(cols, false);
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
        int lastInserted = -1;
        for (int bx = 0; bx < cols-1; bx++) {
            double dxRef = fabs(rowRef[bx+1][channel]-rowRef[bx][channel]);
            double dxDist  = fabs(rowDist[bx+1][channel]-rowDist[bx][channel]);
            float difference = rowRef[bx][channel] - rowDist[bx][channel];
            bool guess_impulse = dxDist>=2*dxRef &&
                abs(difference)>=2*sum_diffs/cols && dxDist>2*sum_dx/(cols-1);
            if (guess_impulse) {
                impulse[bx]=true;
            }
        }
    }
    int lastInserted = -1;
    int impulse_counter = 0;
    for (int bx = 0; bx < cols; bx++) {
        if (impulse[bx]) {
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
    return impulse_counter++;
}


ImpulseStats clean_impulse_lab(const cv::Mat& refLab32,
                                 const cv::Mat& distLab32,
                                 cv::Mat& cleanedLab32)
{
    assert(refLab32.size() == distLab32.size());
    assert(refLab32.type() == CV_32FC3);
    assert(distLab32.type() == CV_32FC3);

    cleanedLab32.create(distLab32.size(), distLab32.type());

    int totalImpulses = 0;

    const int rows = distLab32.rows;
    const int cols = distLab32.cols;

    for (int y = 0; y < rows; ++y) {
        const auto* rowRef  = refLab32.ptr<cv::Vec3f>(y);
        const auto* rowDist = distLab32.ptr<cv::Vec3f>(y);
        auto*       rowOut  = cleanedLab32.ptr<cv::Vec3f>(y);

        int countRow = clean_impulse_row(cols, rowRef, rowDist, rowOut);
        totalImpulses += countRow;
    }

    ImpulseStats stats;
    stats.count = totalImpulses;
    return stats;
}

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
