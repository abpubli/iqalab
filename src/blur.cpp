#include <iqalab/blur.hpp>
#include <opencv2/imgproc.hpp>

namespace iqa::blur
{
    double l_channel_gradient_energy(const cv::Mat& lab,
                                     const cv::Mat& mask)
    {
        CV_Assert(lab.type() == CV_32FC3);
        CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == lab.size()));

        cv::Mat L;
        cv::extractChannel(lab, L, 0);

        cv::Mat L_blur;
        cv::GaussianBlur(L, L_blur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);

        cv::Mat gx, gy;
        cv::Sobel(L_blur, gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(L_blur, gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

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
            return 0.0; // ref płaski – nic do rozmazywania

        double r = E_dist / (E_ref + eps);
        double d = 1.0 - r;

        if (d < 0.0) d = 0.0;
        if (d > 1.5) d = 1.5;
        return d;
    }


// Computes mean squared gradient magnitude for a+b channels in Lab (CV_32FC3).
// If mask is provided (CV_8U, 0/255), the mean is taken only over masked pixels.
double ab_channels_gradient_energy(const cv::Mat& lab,
                                   const cv::Mat& mask)
{
    CV_Assert(lab.type() == CV_32FC3);
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == lab.size()));

    // Extract a and b channels.
    cv::Mat aCh, bCh;
    cv::extractChannel(lab, aCh, 1);
    cv::extractChannel(lab, bCh, 2);

    // Optional Gaussian smoothing to set the observation scale.
    cv::Mat aBlur, bBlur;
    cv::GaussianBlur(aCh, aBlur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(bCh, bBlur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);

    // Sobel gradients for a and b channels.
    cv::Mat agx, agy, bgx, bgy;
    cv::Sobel(aBlur, agx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(aBlur, agy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(bBlur, bgx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(bBlur, bgy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

    // Gradient energy for chroma: |∇a|^2 + |∇b|^2.
    cv::Mat agx2, agy2, bgx2, bgy2, g2;
    cv::multiply(agx, agx, agx2);
    cv::multiply(agy, agy, agy2);
    cv::multiply(bgx, bgx, bgx2);
    cv::multiply(bgy, bgy, bgy2);

    g2 = agx2 + agy2 + bgx2 + bgy2;

    if (!mask.empty())
    {
        // Convert mask to 0/1 float and use it as a weighting map.
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
        // Unmasked case: simple mean over the whole image.
        cv::Scalar meanG2 = cv::mean(g2);
        return static_cast<double>(meanG2[0]);
    }
}

// Relative blur in a+b (chroma) channel pair.
//
// labRef, labDist:
//   - Lab images in CV_32FC3, same size.
// mask:
//   - Optional CV_8U mask (0/255). If empty, whole image is used.
// eps:
//   - Small regularization constant to avoid division by zero.
//
// Returns:
//   - blur measure in [0, ~1.5], where:
//       0   ~ no blur (or reference is flat in chroma),
//       >0  ~ increasing blur strength in a+b.
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

    // If the reference is almost flat in chroma, there is nothing meaningful to blur.
    if (E_ref <= eps)
        return 0.0;

    // r < 1  → blur (energy drop),
    // r ~ 1  → no blur,
    // r > 1  → sharpening or noise (we clamp to zero later).
    double r = E_dist / (E_ref + eps);
    double d = 1.0 - r;

    // Clamp to a reasonable range to avoid exploding values on extreme cases.
    if (d < 0.0)
        d = 0.0;
    if (d > 1.5)
        d = 1.5;

    return d;
}
}
