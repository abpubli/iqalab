#pragma once

#include <opencv2/core.hpp>

namespace iqa {
namespace regions {

// Lightweight regular grid of blocks, no full-size masks are stored.
struct BlockGrid16 {
    cv::Size imageSize; // full image size
    int blockSize = 16; // typically 16
    int blocksX = 0; // number of blocks horizontally
    int blocksY = 0; // number of blocks vertically
};

// Block-level region masks: each block is assigned a single class.
// All pixels inside the block are set to 255 only in the mask that
// corresponds to the chosen class.
struct BlockRegionMasks {
    cv::Mat1b flat;    // 255 where block is classified as flat
    cv::Mat1b mid;     // 255 where block is classified as mid
    cv::Mat1b detail;  // 255 where block is classified as detail
};

// Build block-level masks from pixel-level flat/mid/detail masks.
//
// For each 16x16 block:
//  - count how many pixels belong to flat/mid/detail masks,
//  - normally choose the class with the highest count (majority),
//  - SPECIAL CASE:
//      if flat and detail both occupy a large fraction of the block
//      and mid has small fraction, the block is classified as mid.
//
// Parameters:
//  - minDominantFrac:
//      minimal fraction of pixels for the dominant class to assign
//      it as a block label (otherwise the block stays unclassified).
//  - strongPairFrac:
//      threshold for "large part" of flat and detail in the special
//      case (flat+detail both >= strongPairFrac, mid < strongPairFrac).
BlockRegionMasks make_block_region_masks_from_pixel_masks(
    const BlockGrid16& grid,
    const cv::Mat1b& flatMask,
    const cv::Mat1b& midMask,
    const cv::Mat1b& detailMask,
    double minDominantFrac = 0.5,
    double strongPairFrac  = 0.3
);

// Initialize a regular block grid for given image size.
BlockGrid16 make_block16_grid(const cv::Size& size, int blockSize = 16);

// Get block index for pixel (x,y).
// For blockSize == 16 you can later replace division by shift (x >> 4, y >> 4).
int block_index(const BlockGrid16& g, int x, int y);

// Get rectangle (ROI) for given block index.
cv::Rect block_rect(const BlockGrid16& g, int blockIndex);

} // namespace regions
} // namespace iqa