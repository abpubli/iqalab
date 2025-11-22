#include "iqalab/mse.hpp"

#include "iqalab/mse.hpp"
#include <stdexcept>

namespace iqa::mse
{
double compute_mse_single_channel(const cv::Mat& ref, const cv::Mat& test)
{
  CV_Assert(ref.size() == test.size());
  CV_Assert(ref.channels() == 1);
  CV_Assert(test.channels() == 1);
  CV_Assert(ref.depth() == test.depth());

  cv::Mat ref32, test32;
  if (ref.depth() == CV_32F) {
    ref32 = ref;
    test32 = test;
  } else {
    ref.convertTo(ref32, CV_32F);
    test.convertTo(test32, CV_32F);
  }

  const int rows = ref32.rows;
  const int cols = ref32.cols;

  double sumSq = 0.0;

  for (int y = 0; y < rows; ++y) {
    const float* r = ref32.ptr<float>(y);
    const float* t = test32.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      const double d = static_cast<double>(r[x]) - static_cast<double>(t[x]);
      sumSq += d * d;
    }
  }

  const double N = static_cast<double>(rows) * static_cast<double>(cols);
  return (N > 0.0) ? (sumSq / N) : 0.0;
}

double compute_mse(const cv::Mat& ref, const cv::Mat& test)
{
  CV_Assert(ref.size() == test.size());
  CV_Assert(ref.type() == test.type());
  CV_Assert(ref.channels() == 1 || ref.channels() == 3);

  if (ref.channels() == 1) {
    return compute_mse_single_channel(ref, test);
  }

  // BGR: liczymy MSE jako średnią po kanałach
  std::vector<cv::Mat> refCh, testCh;
  cv::split(ref, refCh);
  cv::split(test, testCh);

  double mseSum = 0.0;
  for (int c = 0; c < 3; ++c) {
    mseSum += compute_mse_single_channel(refCh[c], testCh[c]);
  }
  return mseSum / 3.0;
}

double lab_channel_mse(const cv::Mat& labRef,
                       const cv::Mat& labDist,
                       int channel,
                       const cv::Mat& mask)
{
  CV_Assert(labRef.type() == CV_32FC3);
  CV_Assert(labDist.type() == CV_32FC3);
  CV_Assert(labRef.size() == labDist.size());
  CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == labRef.size()));

  cv::Mat refCh, distCh;
  cv::extractChannel(labRef,  refCh,  channel);
  cv::extractChannel(labDist, distCh, channel);

  cv::Mat diff;
  cv::subtract(refCh, distCh, diff, cv::noArray(), CV_32F);

  cv::Mat diff2;
  cv::multiply(diff, diff, diff2);

  if (!mask.empty())
  {
    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);
    cv::Scalar sumDiff2 = cv::sum(diff2.mul(maskFloat));
    double sumMask = cv::sum(maskFloat)[0];
    if (sumMask <= 0.0)
      return 0.0;
    return static_cast<double>(sumDiff2[0] / sumMask);
  }
  else
  {
    cv::Scalar meanDiff2 = cv::mean(diff2);
    return static_cast<double>(meanDiff2[0]);
  }
}
}
