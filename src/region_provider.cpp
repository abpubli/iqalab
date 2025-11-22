#include "iqalab/region_provider.hpp"

#include <stdexcept>

#include <opencv2/core/mat.hpp>

#include "iqalab/region_masks.hpp"

namespace iqa
{

// PixelwiseRegionProvider
// ------------------------------------------------------------

PixelwiseRegionProvider::PixelwiseRegionProvider(float flatPercentile,
                                                 float detailPercentile)
    : m_flatPercentile(flatPercentile)
    , m_detailPercentile(detailPercentile)
{
    // For now we do not override the internal thresholds used by the
    // existing compute_region_masks() implementation. The percentile
    // values are stored for documentation/debugging and can be wired
    // into region_masks.cpp once it is refactored to accept parameters.
}

RegionMasks PixelwiseRegionProvider::compute_regions(const cv::Mat& labRef) const
{
    // Delegates to the current percentile-based implementation.
    // In the future this can be extended to pass m_flatPercentile /
    // m_detailPercentile down to the region-masks core if needed.
    return compute_region_masks(labRef);
}


// BlockRegionProvider
// ------------------------------------------------------------

BlockRegionProvider::BlockRegionProvider(int blockSize)
    : m_blockSize(blockSize)
{
}

RegionMasks BlockRegionProvider::compute_regions(const cv::Mat& labRef) const
{
    (void)labRef;
    // TODO: Implement block-based segmentation (e.g. 16x16 grid) using
    // block-level gradient statistics in L channel.
    //
    // For now this provider is a stub to make the RegionProvider abstraction
    // explicit without changing behaviour.
    throw std::logic_error(
        "BlockRegionProvider::compute_regions is not implemented yet");
}


// SuperpixelRegionProvider
// ------------------------------------------------------------

SuperpixelRegionProvider::SuperpixelRegionProvider(int   desiredSuperpixels,
                                                   float compactness)
    : m_desiredSuperpixels(desiredSuperpixels)
    , m_compactness(compactness)
{
}

RegionMasks SuperpixelRegionProvider::compute_regions(const cv::Mat& labRef) const
{
    (void)labRef;
    // TODO: Implement superpixel-based segmentation (e.g. SLIC in Lab space)
    // and classify superpixels into flat/mid/detail using region-level
    // gradient/texture statistics.
    //
    // This stub makes the RegionProvider API ready while keeping the current
    // behaviour unchanged in existing tools.
    throw std::logic_error(
        "SuperpixelRegionProvider::compute_regions is not implemented yet");
}


// Factory
// ------------------------------------------------------------

std::unique_ptr<RegionProvider> make_default_region_provider()
{
    // For now we simply return the existing percentile-based provider.
    // This can later be switched to BlockRegionProvider or
    // SuperpixelRegionProvider after experimental evaluation, without
    // touching caller code.
    return std::unique_ptr<RegionProvider>(
        new PixelwiseRegionProvider(/*flatPercentile=*/0.3f,
                                    /*detailPercentile=*/0.7f));
}

} // namespace iqa
