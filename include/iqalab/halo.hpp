#pragma once

#include <opencv2/core/mat.hpp>

namespace iqa::halo
{

// Aggregated halo metrics measured on strong edges in the detail region.
struct HaloMetrics
{
  // Luminance halo (white/black halo).
  double halo_L_strength_detail  = 0.0; // mean relative halo strength in L
  double halo_L_fraction_detail  = 0.0; // fraction of strong edges with L halo
  double halo_L_width_detail     = 0.0; // average halo width in L (pixels)

  // Chromatic halo (rainbow / color fringing).
  double halo_ab_strength_detail = 0.0; // mean absolute chroma halo strength (Lab units)
  double halo_ab_fraction_detail = 0.0; // fraction of strong edges with chromatic halo
  double halo_ab_width_detail    = 0.0; // average halo width in a+b (pixels)
};

// Compute halo metrics on the detail region.
//
// labRef, labDist:
//   - reference and distorted images in Lab32 (CV_32FC3), same size.
// detailMask:
//   - CV_8U mask (0/255) marking detail region (where to search for edges).
//
// The function:
//   - finds strong edges in L within detailMask (percentile-based threshold),
//   - for each edge pixel samples a 1D profile along the normal to the edge,
//   - measures luminance halo (overshoot/undershoot relative to local contrast),
//   - measures chromatic halo (magnitude of a+b deviation),
//   - aggregates per-edge values into HaloMetrics.
HaloMetrics compute_halo_metrics(const cv::Mat& labRef,
                                 const cv::Mat& labDist,
                                 const cv::Mat& detailMask);

} // namespace iqa::halo
