#pragma once
#include <cstddef>

namespace iqa {
void linear_regression(double sumX, double sumY,
                              double sumXX, double sumXY,
                              std::size_t n, double &a_out,
                              double &b_out);
}