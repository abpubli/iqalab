#pragma once

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "iqalab/region_masks.hpp"

namespace iqa
{

// Abstract interface for region segmentation used by IQA feature extractors.
//
// Implementations provide flat/mid/detail masks on a reference image
// in Lab32 (CV_32FC3). Different strategies (pixelwise percentiles,
// block-based, superpixels, etc.) can be plugged in behind this interface.
class RegionProvider
{
public:
    virtual ~RegionProvider() = default;

    // Compute flat/mid/detail masks for the given reference image.
    // The input must be in Lab color space (CV_32FC3), matching the rest
    // of iqalab's internal pipeline.
    virtual RegionMasks compute_regions(const cv::Mat& labRef) const = 0;

    // Optional identifier for logging / debugging / CSV metadata.
    virtual std::string name() const = 0;
};


// Region provider using the existing pixelwise percentile-based segmentation.
//
// For each image independently, gradient energy in L channel is computed
// and thresholded at flatPercentile / detailPercentile, yielding:
//   - ≈ flatPercentile   of pixels as flat,
//   - ≈ (detailP - flatP) as mid,
//   - ≈ (1 - detailP)    as detail.
//
// This is the current baseline implementation behind compute_region_masks().
class PixelwiseRegionProvider final : public RegionProvider
{
public:
    PixelwiseRegionProvider(float flatPercentile   = 0.3f,
                            float detailPercentile = 0.7f);

    RegionMasks compute_regions(const cv::Mat& labRef) const override;

    std::string name() const override { return "pixelwise_percentiles"; }

    float flat_percentile() const   { return m_flatPercentile; }
    float detail_percentile() const { return m_detailPercentile; }

private:
    float m_flatPercentile;
    float m_detailPercentile;
};


// Block-based region provider (grid 16x16 or similar).
//
// The idea is to classify larger spatial blocks instead of individual pixels,
// using block-level gradient statistics. This reduces pixel-level noise and
// naturally aligns with block-based compression artefacts.
//
// For now this is a placeholder; compute_regions() throws, and will be
// implemented once the grid-based scheme is finalized.
class BlockRegionProvider final : public RegionProvider
{
public:
    // blockSize: width/height of the grid blocks in pixels (e.g. 16).
    explicit BlockRegionProvider(int blockSize = 16);

    RegionMasks compute_regions(const cv::Mat& labRef) const override;

    std::string name() const override { return "block_grid"; }

    int block_size() const { return m_blockSize; }

private:
    int m_blockSize;
};


// Superpixel-based region provider (e.g. SLIC in Lab space).
//
// The reference image is segmented into superpixels, then each superpixel
// is classified into flat/mid/detail based on region-level gradient/texture
// statistics. This is expected to be the most perceptually meaningful scheme
// but also the heaviest.
//
// For now this is a placeholder; compute_regions() throws, and will be
// implemented once the SLIC-based pipeline is in place.
class SuperpixelRegionProvider final : public RegionProvider
{
public:
    // desiredSuperpixels: target number of superpixels per image.
    // compactness: SLIC compactness parameter (color vs spatial balance).
    SuperpixelRegionProvider(int desiredSuperpixels = 800,
                             float compactness      = 10.0f);

    RegionMasks compute_regions(const cv::Mat& labRef) const override;

    std::string name() const override { return "superpixel"; }

    int   desired_superpixels() const { return m_desiredSuperpixels; }
    float compactness() const         { return m_compactness; }

private:
    int   m_desiredSuperpixels;
    float m_compactness;
};


// Convenience factory for the default provider used in examples and tools.
//
// Currently this returns a PixelwiseRegionProvider with standard percentiles,
// but it can be switched later to block- or superpixel-based segmentation
// without changing caller code.
std::unique_ptr<RegionProvider> make_default_region_provider();

} // namespace iqa
