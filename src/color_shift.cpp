#include "iqalab/color_shift.hpp"
#include "iqalab/math_utils.hpp"

namespace iqa {

LabShift compute_lab_shift(const cv::Mat& ref,
                           const cv::Mat& dist)
{
  CV_Assert(ref.type()  == CV_32FC3);
  CV_Assert(dist.type() == CV_32FC3);
  CV_Assert(ref.size()  == dist.size());

  const int rows = ref.rows;
  const int cols = ref.cols;

  // L
  double sumLx  = 0.0, sumLy  = 0.0;
  double sumLxx = 0.0, sumLxy = 0.0;
  std::size_t nL = 0;

  // a*
  double sumAx  = 0.0, sumAy  = 0.0;
  double sumAxx = 0.0, sumAxy = 0.0;
  std::size_t nA = 0;

  // b*
  double sumBx  = 0.0, sumBy  = 0.0;
  double sumBxx = 0.0, sumBxy = 0.0;
  std::size_t nB = 0;

  for (int y = 0; y < rows; ++y) {
    const cv::Vec3f* rRow = ref.ptr<cv::Vec3f>(y);
    const cv::Vec3f* dRow = dist.ptr<cv::Vec3f>(y);

    for (int x = 0; x < cols; ++x) {
      const cv::Vec3f& R = rRow[x];
      const cv::Vec3f& D = dRow[x];

      float Lr = R[0], Ld = D[0];
      float ar = R[1], ad = D[1];
      float br = R[2], bd = D[2];

      // L*
      sumLx  += Lr;
      sumLy  += Ld;
      sumLxx += double(Lr) * Lr;
      sumLxy += double(Lr) * Ld;
      ++nL;

      // a*
      sumAx  += ar;
      sumAy  += ad;
      sumAxx += double(ar) * ar;
      sumAxy += double(ar) * ad;
      ++nA;

      // b*
      sumBx  += br;
      sumBy  += bd;
      sumBxx += double(br) * br;
      sumBxy += double(br) * bd;
      ++nB;
    }
  }

  LabShift out;
  linear_regression(sumLx, sumLy, sumLxx, sumLxy, nL, out.a_L, out.b_L);
  linear_regression(sumAx, sumAy, sumAxx, sumAxy, nA, out.a_a, out.b_a);
  linear_regression(sumBx, sumBy, sumBxx, sumBxy, nB, out.a_b, out.b_b);
  return out;
}

} // namespace iqa
