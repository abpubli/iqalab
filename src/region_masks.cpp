#include "iqalab/region_masks.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>

namespace iqa {

static float percentile_from_vector(std::vector<float>& vals, float p)
{
    if (vals.empty()) return 0.0f;
    if (p <= 0.0f) return vals.front();
    if (p >= 1.0f) return vals.back();

    std::sort(vals.begin(), vals.end());
    float idx = p * static_cast<float>(vals.size() - 1);
    auto i  = static_cast<size_t>(idx);
    size_t j  = std::min(i + 1, vals.size() - 1);
    float t   = idx - static_cast<float>(i);
    return (1.0f - t) * vals[i] + t * vals[j];
}

RegionMasks compute_region_masks(const cv::Mat& img,
                               float flatPercentile,
                               float detailPercentile)
{
    CV_Assert(!img.empty());

    cv::Mat gray32;

    if (img.type() == CV_32FC3)
    {
        // Zakładamy Lab32: bierzemy kanał L
        cv::Mat L;
        cv::extractChannel(img, L, 0);
        if (L.type() == CV_32FC1)
        {
            gray32 = L;
        }
        else
        {
            L.convertTo(gray32, CV_32F);
        }
    }
    else if (img.type() == CV_8UC3)
    {
        // Zakładamy BGR8: konwersja do grayscale
        cv::Mat gray8;
        cv::cvtColor(img, gray8, cv::COLOR_BGR2GRAY);
        gray8.convertTo(gray32, CV_32F);
    }
    else if (img.type() == CV_32FC1)
    {
        // Już mamy gray32
        gray32 = img;
    }
    else if (img.type() == CV_8UC1)
    {
        // Szary 8-bit – też można przyjąć
        img.convertTo(gray32, CV_32F);
    }
    else
    {
        CV_Error(cv::Error::StsBadArg,
                 "compute_region_masks: unsupported image type");
    }

    return compute_region_masks32(gray32);
}



RegionMasks compute_region_masks32(const cv::Mat& refL,
                               float flatPercentile,
                               float detailPercentile)
{
    CV_Assert(refL.type() == CV_32F);

    RegionMasks masks;

    // 1) Sobel gradient + magnitude
    cv::Mat gx, gy, mag;
    cv::Sobel(refL, gx, CV_32F, 1, 0, 3);
    cv::Sobel(refL, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    // 2) Light blur so as not to react to individual pixels
    cv::GaussianBlur(mag, masks.gradMag, cv::Size(3, 3), 0.8);

    // 3) Percentiles from gradMag
    cv::Mat tmp = masks.gradMag.reshape(1, masks.gradMag.total());
    std::vector<float> vals(tmp.total());
    std::memcpy(vals.data(), tmp.ptr<float>(), vals.size() * sizeof(float));

    float thrFlat   = percentile_from_vector(vals, flatPercentile);
    float thrDetail = percentile_from_vector(vals, detailPercentile);

    masks.flat.create(refL.size(), CV_8U);
    masks.detail.create(refL.size(), CV_8U);
    masks.mid.create(refL.size(), CV_8U);

    for (int y = 0; y < refL.rows; ++y) {
        const float* grow = masks.gradMag.ptr<float>(y);
        auto* frow = masks.flat.ptr<uchar>(y);
        auto* drow = masks.detail.ptr<uchar>(y);
        auto* mrow = masks.mid.ptr<uchar>(y);

        for (int x = 0; x < refL.cols; ++x) {
            float g = grow[x];

            bool isFlat   = (g <= thrFlat);
            bool isDetail = (g >= thrDetail);

            frow[x] = isFlat   ? 255 : 0;
            drow[x] = isDetail ? 255 : 0;
            mrow[x] = (!isFlat && !isDetail) ? 255 : 0;
        }
    }

    return masks;
}

static void masked_absdiff_stats(const cv::Mat& a,
                                 const cv::Mat& b,
                                 const cv::Mat& mask,
                                 double& meanAbsDiff,
                                 double& p95AbsDiff,
                                 int& count)
{
    CV_Assert(a.type() == CV_32F);
    CV_Assert(b.type() == CV_32F);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(a.size() == b.size());
    CV_Assert(a.size() == mask.size());

    std::vector<float> diffs;
    diffs.reserve(a.rows * a.cols / 4);

    for (int y = 0; y < a.rows; ++y) {
        const auto* ar = a.ptr<float>(y);
        const auto* br = b.ptr<float>(y);
        const auto* mr = mask.ptr<uchar>(y);
        for (int x = 0; x < a.cols; ++x) {
            if (mr[x]) {
                float d = std::fabs(ar[x] - br[x]);
                diffs.push_back(d);
            }
        }
    }

    count = static_cast<int>(diffs.size());
    if (diffs.empty()) {
        meanAbsDiff = 0.0;
        p95AbsDiff  = 0.0;
        return;
    }

    double sum = 0.0;
    for (float v : diffs) sum += v;
    meanAbsDiff = sum / static_cast<double>(diffs.size());

    std::sort(diffs.begin(), diffs.end());
    float idx = 0.95f * static_cast<float>(diffs.size() - 1);
    auto i  = static_cast<size_t>(idx);
    size_t j  = std::min(i + 1, diffs.size() - 1);
    float t   = idx - static_cast<float>(i);
    p95AbsDiff = (1.0f - t) * diffs[i] + t * diffs[j];
}

ImpulseScore score_impulses(const cv::Mat& refL,
                            const cv::Mat& distL,
                            const RegionMasks& masks)
{
    CV_Assert(refL.type() == CV_32F);
    CV_Assert(distL.type() == CV_32F);
    CV_Assert(refL.size() == distL.size());

    ImpulseScore s{};
    masked_absdiff_stats(refL, distL, masks.flat,
                         s.meanOnFlat, s.p95OnFlat, s.countFlat);
    return s;
}

static void masked_gradloss_stats(const cv::Mat& magRef,
                                  const cv::Mat& magDist,
                                  const cv::Mat& mask,
                                  double& meanLoss,
                                  double& p95Loss,
                                  int& count)
{
    CV_Assert(magRef.type() == CV_32F);
    CV_Assert(magDist.type() == CV_32F);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(magRef.size() == magDist.size());
    CV_Assert(magRef.size() == mask.size());

    std::vector<float> losses;
    losses.reserve(magRef.rows * magRef.cols / 4);

    for (int y = 0; y < magRef.rows; ++y) {
        const auto* rrow = magRef.ptr<float>(y);
        const auto* drow = magDist.ptr<float>(y);
        const auto* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < magRef.cols; ++x) {
            if (mrow[x]) {
                float loss = rrow[x] - drow[x];
                if (loss > 0.0f) {
                    losses.push_back(loss);
                }
            }
        }
    }

    count = static_cast<int>(losses.size());
    if (losses.empty()) {
        meanLoss = 0.0;
        p95Loss  = 0.0;
        return;
    }

    double sum = 0.0;
    for (float v : losses) sum += v;
    meanLoss = sum / static_cast<double>(losses.size());

    std::sort(losses.begin(), losses.end());
    float idx = 0.95f * static_cast<float>(losses.size() - 1);
    auto i  = static_cast<size_t>(idx);
    size_t j  = std::min(i + 1, losses.size() - 1);
    float t   = idx - static_cast<float>(i);
    p95Loss = (1.0f - t) * losses[i] + t * losses[j];
}

BlurScore score_blur(const cv::Mat& refL,
                     const cv::Mat& distL,
                     const RegionMasks& masks)
{
    CV_Assert(refL.type() == CV_32F);
    CV_Assert(distL.type() == CV_32F);
    CV_Assert(refL.size() == distL.size());

    // Grad ref już mamy w masks.gradMag
    cv::Mat gxD, gyD, magD;
    cv::Sobel(distL, gxD, CV_32F, 1, 0, 3);
    cv::Sobel(distL, gyD, CV_32F, 0, 1, 3);
    cv::magnitude(gxD, gyD, magD);
    cv::GaussianBlur(magD, magD, cv::Size(3, 3), 0.8);

    BlurScore s{};
    masked_gradloss_stats(masks.gradMag, magD, masks.detail,
                          s.meanLossOnDetail, s.p95LossOnDetail, s.countDetail);
    return s;
}

} // namespace iqa
