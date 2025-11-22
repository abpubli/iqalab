#include "iqalab/blur.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace iqa::blur
{

// Helper: mean squared gradient magnitude for L channel in Lab (CV_32FC3).
// If mask is provided (CV_8U, 0/255), the mean is taken only over masked pixels.
double l_channel_gradient_energy(const cv::Mat& lab,
                                        const cv::Mat& mask)
{
    CV_Assert(lab.type() == CV_32FC3);
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == lab.size()));

    cv::Mat lCh;
    cv::extractChannel(lab, lCh, 0);

    // Gaussian smoothing to set the observation scale.
    cv::Mat lBlur;
    cv::GaussianBlur(lCh, lBlur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);

    // Sobel gradients.
    cv::Mat gx, gy;
    cv::Sobel(lBlur, gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(lBlur, gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

    cv::Mat gx2, gy2, g2;
    cv::multiply(gx, gx, gx2);
    cv::multiply(gy, gy, gy2);
    g2 = gx2 + gy2;

    if (!mask.empty())
    {
        cv::Mat maskFloat;
        mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

        cv::Scalar sumG2 = cv::sum(g2.mul(maskFloat));
        double sumMask = cv::sum(maskFloat)[0];

        if (sumMask <= 0.0)
            return 0.0;

        return static_cast<double>(sumG2[0] / sumMask);
    }
    else
    {
        cv::Scalar meanG2 = cv::mean(g2);
        return static_cast<double>(meanG2[0]);
    }
}

// Relative blur in L channel.
double relative_blur_L(const cv::Mat& labRef,
                       const cv::Mat& labDist,
                       const cv::Mat& mask,
                       double eps)
{
    CV_Assert(labRef.type() == CV_32FC3);
    CV_Assert(labDist.type() == CV_32FC3);
    CV_Assert(labRef.size() == labDist.size());
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == labRef.size()));

    double E_ref  = l_channel_gradient_energy(labRef,  mask);
    double E_dist = l_channel_gradient_energy(labDist, mask);

    if (E_ref <= eps)
        return 0.0;

    double r = E_dist / (E_ref + eps);
    double d = 1.0 - r;

    if (d < 0.0) d = 0.0;
    if (d > 1.5) d = 1.5;

    return d;
}

// Helper: mean squared gradient magnitude for a+b channels in Lab (CV_32FC3).
// If mask is provided (CV_8U, 0/255), the mean is taken only over masked pixels.
double ab_channels_gradient_energy(const cv::Mat& lab,
                                          const cv::Mat& mask)
{
    CV_Assert(lab.type() == CV_32FC3);
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == lab.size()));

    cv::Mat aCh, bCh;
    cv::extractChannel(lab, aCh, 1);
    cv::extractChannel(lab, bCh, 2);

    cv::Mat aBlur, bBlur;
    cv::GaussianBlur(aCh, aBlur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(bCh, bBlur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);

    cv::Mat agx, agy, bgx, bgy;
    cv::Sobel(aBlur, agx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(aBlur, agy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(bBlur, bgx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(bBlur, bgy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

    cv::Mat agx2, agy2, bgx2, bgy2, g2;
    cv::multiply(agx, agx, agx2);
    cv::multiply(agy, agy, agy2);
    cv::multiply(bgx, bgx, bgx2);
    cv::multiply(bgy, bgy, bgy2);

    g2 = agx2 + agy2 + bgx2 + bgy2;

    if (!mask.empty())
    {
        cv::Mat maskFloat;
        mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

        cv::Scalar sumG2 = cv::sum(g2.mul(maskFloat));
        double sumMask = cv::sum(maskFloat)[0];

        if (sumMask <= 0.0)
            return 0.0;

        return static_cast<double>(sumG2[0] / sumMask);
    }
    else
    {
        cv::Scalar meanG2 = cv::mean(g2);
        return static_cast<double>(meanG2[0]);
    }
}

// Relative blur in a+b (chroma) channel pair.
double relative_blur_ab(const cv::Mat& labRef,
                        const cv::Mat& labDist,
                        const cv::Mat& mask,
                        double eps)
{
    CV_Assert(labRef.type() == CV_32FC3);
    CV_Assert(labDist.type() == CV_32FC3);
    CV_Assert(labRef.size() == labDist.size());
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == labRef.size()));

    double E_ref  = ab_channels_gradient_energy(labRef,  mask);
    double E_dist = ab_channels_gradient_energy(labDist, mask);

    if (E_ref <= eps)
        return 0.0;

    double r = E_dist / (E_ref + eps);
    double d = 1.0 - r;

    if (d < 0.0) d = 0.0;
    if (d > 1.5) d = 1.5;

    return d;
}

// Relative sharpening / high-frequency increase in L channel.
double relative_sharp_L(const cv::Mat& labRef,
                        const cv::Mat& labDist,
                        const cv::Mat& mask,
                        double eps)
{
    CV_Assert(labRef.type() == CV_32FC3);
    CV_Assert(labDist.type() == CV_32FC3);
    CV_Assert(labRef.size() == labDist.size());
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == labRef.size()));

    double E_ref  = l_channel_gradient_energy(labRef,  mask);
    double E_dist = l_channel_gradient_energy(labDist, mask);

    if (E_ref <= eps)
        return 0.0;

    double r = E_dist / (E_ref + eps);
    double s = r - 1.0;

    if (s < 0.0) s = 0.0;
    if (s > 1.5) s = 1.5;

    return s;
}

// Relative sharpening / high-frequency increase in a+b (chroma) channel pair.
double relative_sharp_ab(const cv::Mat& labRef,
                         const cv::Mat& labDist,
                         const cv::Mat& mask,
                         double eps)
{
    CV_Assert(labRef.type() == CV_32FC3);
    CV_Assert(labDist.type() == CV_32FC3);
    CV_Assert(labRef.size() == labDist.size());
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == labRef.size()));

    double E_ref  = ab_channels_gradient_energy(labRef,  mask);
    double E_dist = ab_channels_gradient_energy(labDist, mask);

    if (E_ref <= eps)
        return 0.0;

    double r = E_dist / (E_ref + eps);
    double s = r - 1.0;

    if (s < 0.0) s = 0.0;
    if (s > 1.5) s = 1.5;

    return s;
}

} // namespace iqa::blur
