#include "iqalab/halo.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace iqa::halo
{

namespace
{

struct HaloParams
{
    // Edge selection.
    double edgePercentile = 0.85; // percentile of gradient magnitude in detail region
    double minContrastL   = 5.0;  // minimum luminance contrast to consider an edge (Lab units)

    // Profile sampling.
    int    profileRadius  = 4;    // samples from -R..+R along the normal
    double profileStep    = 1.0;  // step in pixels (integer sampling here)

    // Luminance halo thresholds.
    double epsHalo        = 1e-3; // small epsilon for contrast normalization
    double haloLThreshold = 0.10; // relative halo threshold (fraction of local contrast)

    // Chromatic halo thresholds.
    double haloAbThreshold = 2.0; // absolute chroma difference threshold (Lab units)
};

// Compute gradient in L channel (Sobel 3x3 after small Gaussian blur).
// Outputs:
//   gx, gy: CV_32F, same size as lab
//   gradMag: CV_32F, gradient magnitude
void compute_L_gradients(const cv::Mat& lab,
                         cv::Mat& gx,
                         cv::Mat& gy,
                         cv::Mat& gradMag)
{
    CV_Assert(lab.type() == CV_32FC3);

    cv::Mat lCh;
    cv::extractChannel(lab, lCh, 0);

    cv::Mat lBlur;
    cv::GaussianBlur(lCh, lBlur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);

    cv::Sobel(lBlur, gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(lBlur, gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

    cv::magnitude(gx, gy, gradMag);
}

// Compute percentile threshold of gradient magnitude inside detailMask.
float compute_edge_threshold(const cv::Mat& gradMag,
                             const cv::Mat& detailMask,
                             double edgePercentile)
{
    CV_Assert(gradMag.type() == CV_32F);
    CV_Assert(detailMask.type() == CV_8U);
    CV_Assert(gradMag.size() == detailMask.size());

    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(gradMag.total() / 10)); // rough

    const int rows = gradMag.rows;
    const int cols = gradMag.cols;

    for (int y = 0; y < rows; ++y)
    {
        const float* gRow = gradMag.ptr<float>(y);
        const uchar* mRow = detailMask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x)
        {
            if (mRow[x] != 0)
            {
                float v = gRow[x];
                if (v > 0.0f)
                    values.push_back(v);
            }
        }
    }

    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());

    double p = std::clamp(edgePercentile, 0.0, 1.0);
    std::size_t idx = static_cast<std::size_t>(p * (values.size() - 1));

    return values[idx];
}

// Helper: sample a value from a single-channel CV_32F image at float coords,
// using nearest-neighbour (fast and sufficient for halo estimation).
float sample_nn(const cv::Mat& img, float xf, float yf)
{
    int x = static_cast<int>(std::round(xf));
    int y = static_cast<int>(std::round(yf));
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
        return img.at<float>(std::clamp(y, 0, img.rows - 1),
                             std::clamp(x, 0, img.cols - 1));
    return img.at<float>(y, x);
}

// Helper: check if integer coordinates are inside the image.
bool in_bounds(const cv::Mat& img, int x, int y)
{
    return x >= 0 && x < img.cols && y >= 0 && y < img.rows;
}

} // anonymous namespace

HaloMetrics compute_halo_metrics(const cv::Mat& labRef,
                                 const cv::Mat& labDist,
                                 const cv::Mat& detailMask)
{
    CV_Assert(labRef.type() == CV_32FC3);
    CV_Assert(labDist.type() == CV_32FC3);
    CV_Assert(labRef.size() == labDist.size());
    CV_Assert(detailMask.type() == CV_8U);
    CV_Assert(detailMask.size() == labRef.size());

    HaloParams params;
    HaloMetrics out;

    // Extract L,a,b channels.
    cv::Mat L_ref, a_ref, b_ref;
    cv::Mat L_dist, a_dist, b_dist;
    cv::extractChannel(labRef,  L_ref,  0);
    cv::extractChannel(labRef,  a_ref,  1);
    cv::extractChannel(labRef,  b_ref,  2);
    cv::extractChannel(labDist, L_dist, 0);
    cv::extractChannel(labDist, a_dist, 1);
    cv::extractChannel(labDist, b_dist, 2);

    // Compute L gradients.
    cv::Mat gx, gy, gradMag;
    compute_L_gradients(labRef, gx, gy, gradMag);

    // Edge threshold in detail region.
    float edgeThresh = compute_edge_threshold(gradMag, detailMask,
                                              params.edgePercentile);
    if (edgeThresh <= 0.0f)
    {
        // No usable edges -> return zeros.
        return out;
    }

    const int rows = labRef.rows;
    const int cols = labRef.cols;

    const int R = params.profileRadius;
    const double step = params.profileStep;

    std::size_t totalEdgePoints = 0;
    std::size_t haloLPoints     = 0;
    std::size_t haloAbPoints    = 0;

    double sumHaloLStrength  = 0.0;
    double sumHaloLWidth     = 0.0;
    double sumHaloAbStrength = 0.0;
    double sumHaloAbWidth    = 0.0;

    for (int y = 0; y < rows; ++y)
    {
        const uchar* mRow   = detailMask.ptr<uchar>(y);
        const float* gRow   = gradMag.ptr<float>(y);
        const float* gxRow  = gx.ptr<float>(y);
        const float* gyRow  = gy.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
        {
            if (mRow[x] == 0)
                continue;

            float g = gRow[x];
            if (g < edgeThresh)
                continue;

            // Edge candidate.
            ++totalEdgePoints;

            float gxv = gxRow[x];
            float gyv = gyRow[x];

            if (gxv == 0.0f && gyv == 0.0f)
                continue;

            // Normalised normal vector (from dark to bright side, corrected later).
            double nx = gxv / g;
            double ny = gyv / g;

            // Sample L_ref on both sides to determine orientation and contrast.
            // We use small windows around the extremes of the profile.
            double darkSum = 0.0;
            double brightSum = 0.0;
            int darkCount = 0;
            int brightCount = 0;

            for (int t = -R; t <= -2; ++t)
            {
                double xf = static_cast<double>(x) + t * nx * step;
                double yf = static_cast<double>(y) + t * ny * step;
                float Lv = sample_nn(L_ref, static_cast<float>(xf), static_cast<float>(yf));
                darkSum += Lv;
                ++darkCount;
            }
            for (int t = 2; t <= R; ++t)
            {
                double xf = static_cast<double>(x) + t * nx * step;
                double yf = static_cast<double>(y) + t * ny * step;
                float Lv = sample_nn(L_ref, static_cast<float>(xf), static_cast<float>(yf));
                brightSum += Lv;
                ++brightCount;
            }

            if (darkCount == 0 || brightCount == 0)
                continue;

            double darkMean   = darkSum   / darkCount;
            double brightMean = brightSum / brightCount;

            // Ensure orientation: t > 0 goes from dark to bright.
            if (brightMean < darkMean)
            {
                std::swap(darkMean, brightMean);
                nx = -nx;
                ny = -ny;
            }

            double contrastL = brightMean - darkMean;
            if (contrastL < params.minContrastL)
                continue;

            // Now sample full profiles for L and ab.
            double maxOvershootL  = 0.0; // bright side
            double maxUndershootL = 0.0; // dark side (dist darker than ref)

            double maxChromaDev = 0.0;   // absolute chroma deviation

            int haloLPixelsWidth  = 0;
            int haloAbPixelsWidth = 0;

            for (int t = -R; t <= R; ++t)
            {
                double xf = static_cast<double>(x) + t * nx * step;
                double yf = static_cast<double>(y) + t * ny * step;

                float Lr = sample_nn(L_ref,  static_cast<float>(xf), static_cast<float>(yf));
                float Ld = sample_nn(L_dist, static_cast<float>(xf), static_cast<float>(yf));
                float ar = sample_nn(a_ref,  static_cast<float>(xf), static_cast<float>(yf));
                float ad = sample_nn(a_dist, static_cast<float>(xf), static_cast<float>(yf));
                float br = sample_nn(b_ref,  static_cast<float>(xf), static_cast<float>(yf));
                float bd = sample_nn(b_dist, static_cast<float>(xf), static_cast<float>(yf));

                double dL = static_cast<double>(Ld - Lr);

                if (t > 0)
                {
                    // bright side overshoot
                    if (dL > maxOvershootL)
                        maxOvershootL = dL;
                }
                else if (t < 0)
                {
                    // dark side undershoot (dist darker than ref)
                    double undershoot = static_cast<double>(Lr - Ld);
                    if (undershoot > maxUndershootL)
                        maxUndershootL = undershoot;
                }

                double da = static_cast<double>(ad - ar);
                double db = static_cast<double>(bd - br);
                double dC = std::sqrt(da * da + db * db);

                if (dC > maxChromaDev)
                    maxChromaDev = dC;
            }

            // L halo strength, relative to contrast.
            double denom = contrastL + params.epsHalo;
            double haloLPoint = std::max(maxOvershootL, maxUndershootL) / denom;

            bool hasLHalo  = (haloLPoint >= params.haloLThreshold);
            bool hasAbHalo = (maxChromaDev >= params.haloAbThreshold);

            // Compute widths only if we have halo in the corresponding domain.
            if (hasLHalo || hasAbHalo)
            {
                for (int t = -R; t <= R; ++t)
                {
                    double xf = static_cast<double>(x) + t * nx * step;
                    double yf = static_cast<double>(y) + t * ny * step;

                    float Lr = sample_nn(L_ref,  static_cast<float>(xf), static_cast<float>(yf));
                    float Ld = sample_nn(L_dist, static_cast<float>(xf), static_cast<float>(yf));
                    float ar = sample_nn(a_ref,  static_cast<float>(xf), static_cast<float>(yf));
                    float ad = sample_nn(a_dist, static_cast<float>(xf), static_cast<float>(yf));
                    float br = sample_nn(b_ref,  static_cast<float>(xf), static_cast<float>(yf));
                    float bd = sample_nn(b_dist, static_cast<float>(xf), static_cast<float>(yf));

                    double dL = std::abs(static_cast<double>(Ld - Lr));
                    double da = static_cast<double>(ad - ar);
                    double db = static_cast<double>(bd - br);
                    double dC = std::sqrt(da * da + db * db);

                    if (hasLHalo)
                    {
                        if (dL >= params.haloLThreshold * contrastL)
                            ++haloLPixelsWidth;
                    }

                    if (hasAbHalo)
                    {
                        if (dC >= params.haloAbThreshold)
                            ++haloAbPixelsWidth;
                    }
                }
            }

            // Accumulate statistics.
            if (hasLHalo)
            {
                ++haloLPoints;
                sumHaloLStrength += haloLPoint;
                sumHaloLWidth    += haloLPixelsWidth * params.profileStep;
            }

            if (hasAbHalo)
            {
                ++haloAbPoints;
                sumHaloAbStrength += maxChromaDev;
                sumHaloAbWidth    += haloAbPixelsWidth * params.profileStep;
            }
        }
    }

    if (totalEdgePoints == 0)
        return out;

    // L halo aggregation.
    out.halo_L_fraction_detail =
        (totalEdgePoints > 0)
            ? static_cast<double>(haloLPoints) / static_cast<double>(totalEdgePoints)
            : 0.0;

    out.halo_L_strength_detail =
        (haloLPoints > 0)
            ? sumHaloLStrength / static_cast<double>(haloLPoints)
            : 0.0;

    out.halo_L_width_detail =
        (haloLPoints > 0)
            ? sumHaloLWidth / static_cast<double>(haloLPoints)
            : 0.0;

    // Chromatic halo aggregation.
    out.halo_ab_fraction_detail =
        (totalEdgePoints > 0)
            ? static_cast<double>(haloAbPoints) / static_cast<double>(totalEdgePoints)
            : 0.0;

    out.halo_ab_strength_detail =
        (haloAbPoints > 0)
            ? sumHaloAbStrength / static_cast<double>(haloAbPoints)
            : 0.0;

    out.halo_ab_width_detail =
        (haloAbPoints > 0)
            ? sumHaloAbWidth / static_cast<double>(haloAbPoints)
            : 0.0;

    return out;
}

} // namespace iqa::halo
