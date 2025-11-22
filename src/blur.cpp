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
}
