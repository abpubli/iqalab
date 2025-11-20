#include "iqalab/math_utils.hpp"

#include <cmath>
#include <cstddef>

namespace iqa {
void linear_regression(const double sumX,
                          const double sumY,
                          const double sumXX,
                          const double sumXY,
                          const std::size_t n,
                          double& a_out,
                          double& b_out)
{
  if (n == 0) {
    a_out = 1.0;
    b_out = 0.0;
    return;
  }

  auto n_d = static_cast<double>(n);
  double den = n_d * sumXX - sumX * sumX;

  if (std::fabs(den) < 1e-12) {
    // almost no variability in X – degeneration
    a_out = 1.0;
    b_out = (sumY / n_d) - a_out * (sumX / n_d);
    return;
  }

  a_out = (n_d * sumXY - sumX * sumY) / den;
  b_out = (sumY - a_out * sumX) / n_d;
}
}