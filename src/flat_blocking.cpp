#include "iqalab/iqalab.hpp"

#include <deque>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace iqa {
// for a single line – fills the mask 0/1 (0/255)
void analyzeFlatBlocksRow(
    int cols,
    const cv::Vec3f* rowRef,
    const cv::Vec3f* rowDist,
    uchar* maskRow  // pointer to flatMask.ptr<uchar>(y)
) {
    std::vector<bool> inBlock(cols, false);
    int W = 8;
    float diffThr = 1;
    float refThr  = 1;
    const float flatDxThr = 0.5f;

    for (int channel = 0; channel < 3; channel++) {
        std::deque<float> buf, bufRef;
        buf.clear();
        bufRef.clear();
        bool beginDirty = false;

        for (int bx = 0; bx < cols; bx++) {
            buf.push_back(rowDist[bx][channel]);
            bufRef.push_back(rowRef[bx][channel]);

            if ((int)buf.size() > W)     buf.pop_front();
            if ((int)bufRef.size() > W)  bufRef.pop_front();

            float mn    = *std::min_element(buf.begin(), buf.end());
            float mx    = *std::max_element(buf.begin(), buf.end());
            float dx    = mx - mn;

            float mnRef = *std::min_element(bufRef.begin(), bufRef.end());
            float mxRef = *std::max_element(bufRef.begin(), bufRef.end());
            float dxRef = mxRef - mnRef;

            double difference = std::fabs(rowDist[bx][channel] - rowRef[bx][channel]);

            // Criterion:
            // - dx == 0  -> window in dist completely flat
            // - difference >= diffThr or dxRef > refThr
            //   => ref was not that flat or the pixel changed significantly
            if (dx <= flatDxThr && buf.size() == W &&
                (difference >= diffThr || dxRef > refThr)) {

                inBlock[bx] = true;
                if (beginDirty) {
                    for (int i = 0; i < (int)buf.size() - 1; ++i)
                        inBlock[bx - i - 1] = true;
                    beginDirty = false;
                }
            } else {
                beginDirty = true;
            }
        }
    }

    // save the line mask
    for (int bx = 0; bx < cols; ++bx) {
        maskRow[bx] = inBlock[bx] ? 255 : 0;
    }
}

struct Run {
    int y;
    int x0;
    int x1; // [x0, x1) – end not included
};

struct Region {
    int x0, x1;
    int y0, y1;
    int area;     // number of mask pixels=1 in the region
};

std::vector<Region> buildRegionsFromMask(const cv::Mat& mask)
{
    CV_Assert(mask.type() == CV_8U);

    std::vector<Region> finished;
    std::vector<Region> active; // regions “continuing” from the previous row

    const int rows = mask.rows;
    const int cols = mask.cols;

    for (int y = 0; y < rows; ++y) {
        const uchar* row = mask.ptr<uchar>(y);

        // 1. Build runes (strings) for this line
        std::vector<Run> runs;
        bool inRun = false;
        int runStart = 0;

        for (int x = 0; x < cols; ++x) {
            if (row[x]) {
                if (!inRun) {
                    inRun = true;
                    runStart = x;
                }
            } else {
                if (inRun) {
                    runs.push_back(Run{y, runStart, x});
                    inRun = false;
                }
            }
        }
        if (inRun) {
            runs.push_back(Run{y, runStart, cols});
        }

        // 2. Prepare a new list of active regions for this row
        std::vector<Region> newActive;

        // for each run, try to find the region from the previous line
        // with which it intersects horizontally
        std::vector<bool> runAssigned(runs.size(), false);

        for (Region& reg : active) {
            bool extended = false;

            for (size_t i = 0; i < runs.size(); ++i) {
                if (runAssigned[i]) continue;

                const Run& r = runs[i];
                // poziome nakładanie się [x0,x1)
                bool overlap =
                    !(r.x1 <= reg.x0 || r.x0 >= reg.x1);

                if (overlap && r.y == reg.y1 + 1) {
                    // rozszerzamy region
                    reg.x0 = std::min(reg.x0, r.x0);
                    reg.x1 = std::max(reg.x1, r.x1);
                    reg.y1 = r.y;
                    reg.area += (r.x1 - r.x0);
                    runAssigned[i] = true;
                    extended = true;
                    // NOTE: here you can additionally support the case of multiple runes merging into one region
                }
            }

            if (extended) {
                newActive.push_back(reg);
            } else {
                // region completed – move to finished
                finished.push_back(reg);
            }
        }

        // 3. Runes that have not been assigned to any region – new regions are launched
        for (size_t i = 0; i < runs.size(); ++i) {
            if (runAssigned[i]) continue;
            const Run& r = runs[i];
            Region reg;
            reg.x0   = r.x0;
            reg.x1   = r.x1;
            reg.y0   = r.y;
            reg.y1   = r.y;
            reg.area = (r.x1 - r.x0);
            newActive.push_back(reg);
        }

        active.swap(newActive);
    }

    // everything that has been active after the last line – we also end it
    for (Region& reg : active) {
        finished.push_back(reg);
    }
    return finished;
}

cv::Mat flat_blocking_to_mask(const cv::Mat& refBGR, const cv::Mat& distBGR) {
    cv::Mat refBGR32;
    cv::Mat distBGR32;
    cv::Mat ref;
    cv::Mat dist;
    refBGR.convertTo(refBGR32, CV_32FC3, 1.0/255.0);
    cv::cvtColor(refBGR32, ref, cv::COLOR_BGR2Lab);
    distBGR.convertTo(distBGR32, CV_32FC3, 1.0/255.0);
    cv::cvtColor(distBGR32, dist, cv::COLOR_BGR2Lab);

    cv::Mat flatMask(dist.rows, dist.cols, CV_8U, cv::Scalar(0));
    for (int y = 0; y < dist.rows; ++y) {
        const cv::Vec3f* rowRef  = ref.ptr<cv::Vec3f>(y);
        const cv::Vec3f* rowDist = dist.ptr<cv::Vec3f>(y);
        uchar* maskRow = flatMask.ptr<uchar>(y);
        analyzeFlatBlocksRow(dist.cols, rowRef, rowDist, maskRow);
    }
    cv::Mat finalMask(flatMask.size(), CV_8U, cv::Scalar(0));

    const float maxL = 100.0f;
    const float T_diff      = 0.12f * maxL;  // ~12 L*
    const float T_refDetail = 0.03f * maxL;  // ~3 L*
    const float T_flat      = 0.2f * maxL;  // ~1 L*
    const float minRatio    = 0.3f;          // min. fill in bbox
    const int   minArea     = 64;            // cut off all the small stuff

    std::vector<Region> regions = buildRegionsFromMask(flatMask);
    std::cout << regions.size() << std::endl;

    for (const Region& reg : regions) {
        int w = reg.x1 - reg.x0;
        int h = reg.y1 - reg.y0;
        if (w <= 0 || h <= 0) continue;
        if (reg.area < minArea) continue;
        if (cv::min(reg.x1 - reg.x0, reg.y1 - reg.y0)<4) continue;

        int pixelsBox = w * h;
        float ratioMask = static_cast<float>(reg.area) / static_cast<float>(pixelsBox);

        double sumDiff = 0.0;
        double sumRef  = 0.0, sumRef2  = 0.0;
        double sumDist = 0.0, sumDist2 = 0.0;
        int count = 0;

        for (int y = reg.y0; y <= reg.y1; ++y) {
            const uchar* mrow = flatMask.ptr<uchar>(y);
            const cv::Vec3f* rrow = ref.ptr<cv::Vec3f>(y);
            const cv::Vec3f* drow = dist.ptr<cv::Vec3f>(y);

            for (int x = reg.x0; x <= reg.x1; ++x) {
                if (!mrow[x]) continue; // we only count statistics where mask=1

                const cv::Vec3f& rv = rrow[x];
                const cv::Vec3f& dv = drow[x];

                float rL = rv[0];
                float dL = dv[0];
                float diff = std::fabs(dL - rL);

                sumDiff += diff;
                sumRef  += rL;
                sumRef2 += rL * rL;
                sumDist += dL;
                sumDist2+= dL * dL;
                ++count;
            }
        }

        if (count == 0) continue;

        double meanDiff = sumDiff / count;

        double meanRef  = sumRef / count;
        double meanRef2 = sumRef2 / count;
        double varRef   = std::max(0.0, meanRef2 - meanRef*meanRef);
        double stdRef   = std::sqrt(varRef);

        double meanDist  = sumDist / count;
        double meanDist2 = sumDist2 / count;
        double varDist   = std::max(0.0, meanDist2 - meanDist*meanDist);
        double stdDist   = std::sqrt(varDist);

        bool isTransmissionBlock =
            (ratioMask >= minRatio) &&
            (meanDiff  >= T_diff) &&
            (stdRef    >= T_refDetail) &&
            (stdDist   <= T_flat);

        if (!isTransmissionBlock)
            continue;

        // select region in finalMask
        for (int y = reg.y0; y <= reg.y1; ++y) {
            uchar* frow = finalMask.ptr<uchar>(y);
            for (int x = reg.x0; x <= reg.x1; ++x) {
                if (flatMask.at<uchar>(y,x)) {
                    frow[x] = 255;
                }
            }
        }
    }
    return finalMask;
}
}