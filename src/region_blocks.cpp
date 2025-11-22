#include <algorithm>
#include <opencv2/core.hpp>

#include <iqalab/region_blocks.hpp>

namespace iqa {
namespace regions {

BlockGrid16 make_block16_grid(const cv::Size& size, int blockSize)
{
  BlockGrid16 g;
  g.imageSize = size;
  g.blockSize = blockSize;
  // Round up: (n + blockSize - 1) / blockSize
  g.blocksX = (size.width  + blockSize - 1) / blockSize;
  g.blocksY = (size.height + blockSize - 1) / blockSize;

  return g;
}

int block_index(const BlockGrid16& g, int x, int y)
{
  // Assumes (x, y) are inside image bounds.
  const int bx = x / g.blockSize; // for blockSize==16: x >> 4
  const int by = y / g.blockSize; // for blockSize==16: y >> 4
  return by * g.blocksX + bx;
}

cv::Rect block_rect(const BlockGrid16& g, int blockIndex)
{
  const int bx = blockIndex % g.blocksX;
  const int by = blockIndex / g.blocksX;
  const int x0 = bx * g.blockSize;
  const int y0 = by * g.blockSize;

  const int w = std::min(g.blockSize, g.imageSize.width  - x0);
  const int h = std::min(g.blockSize, g.imageSize.height - y0);

  return cv::Rect(x0, y0, w, h);
}


BlockRegionMasks make_block_region_masks_from_pixel_masks(
    const BlockGrid16& grid,
    const cv::Mat1b& flatMask,
    const cv::Mat1b& midMask,
    const cv::Mat1b& detailMask,
    double minDominantFrac,
    double strongPairFrac
)
{
    // Sanity checks: all masks must match the grid image size.
    CV_Assert(flatMask.size()   == grid.imageSize);
    CV_Assert(midMask.size()    == grid.imageSize);
    CV_Assert(detailMask.size() == grid.imageSize);

    CV_Assert(flatMask.type()   == CV_8UC1);
    CV_Assert(midMask.type()    == CV_8UC1);
    CV_Assert(detailMask.type() == CV_8UC1);

    BlockRegionMasks out;
    out.flat   = cv::Mat1b(grid.imageSize, uchar(0));
    out.mid    = cv::Mat1b(grid.imageSize, uchar(0));
    out.detail = cv::Mat1b(grid.imageSize, uchar(0));

    const int totalBlocks = grid.blocksX * grid.blocksY;

    for (int blockIdx = 0; blockIdx < totalBlocks; ++blockIdx) {
        const cv::Rect r = block_rect(grid, blockIdx);
        if (r.width <= 0 || r.height <= 0)
            continue;

        const int area = r.width * r.height;

        int flatCount   = 0;
        int midCount    = 0;
        int detailCount = 0;

        // Count pixels belonging to each class in this block.
        for (int y = r.y; y < r.y + r.height; ++y) {
            const uchar* fRow = flatMask.ptr<uchar>(y);
            const uchar* mRow = midMask.ptr<uchar>(y);
            const uchar* dRow = detailMask.ptr<uchar>(y);

            for (int x = r.x; x < r.x + r.width; ++x) {
                if (fRow[x] == 255)
                    ++flatCount;
                if (mRow[x] == 255)
                    ++midCount;
                if (dRow[x] == 255)
                    ++detailCount;
            }
        }

        if (area <= 0)
            continue;

        const double flatFrac   = static_cast<double>(flatCount)   / static_cast<double>(area);
        const double midFrac    = static_cast<double>(midCount)    / static_cast<double>(area);
        const double detailFrac = static_cast<double>(detailCount) / static_cast<double>(area);

        // Decide which class the block belongs to.
        enum class BlockClass { None, Flat, Mid, Detail };
        BlockClass cls = BlockClass::None;

        // SPECIAL CASE:
        // If both flat and detail have large fraction, and mid is small,
        // classify the block as "mid".
        //
        // "Large part" is controlled by strongPairFrac (e.g. 0.3),
        // "small mid" means midFrac < strongPairFrac.
        if (flatFrac   >= strongPairFrac &&
            detailFrac >= strongPairFrac &&
            midFrac    <  strongPairFrac)
        {
            cls = BlockClass::Mid;
        }
        else {
            // Normal majority decision:
            // choose the class with the highest count, but only if its
            // fraction is at least minDominantFrac.
            int maxCount = flatCount;
            BlockClass maxClass = BlockClass::Flat;

            if (midCount > maxCount) {
                maxCount = midCount;
                maxClass = BlockClass::Mid;
            }
            if (detailCount > maxCount) {
                maxCount = detailCount;
                maxClass = BlockClass::Detail;
            }

            const double maxFrac = static_cast<double>(maxCount) / static_cast<double>(area);

            if (maxFrac >= minDominantFrac) {
                cls = maxClass;
            } else {
                cls = BlockClass::None;  // block stays unclassified
            }
        }

        // Write the chosen class into block-sized region in the output masks.
        cv::Mat1b flatROI   = out.flat(r);
        cv::Mat1b midROI    = out.mid(r);
        cv::Mat1b detailROI = out.detail(r);

        switch (cls) {
        case BlockClass::Flat:
            flatROI.setTo(uchar(255));
            midROI.setTo(uchar(0));
            detailROI.setTo(uchar(0));
            break;
        case BlockClass::Mid:
            flatROI.setTo(uchar(0));
            midROI.setTo(uchar(255));
            detailROI.setTo(uchar(0));
            break;
        case BlockClass::Detail:
            flatROI.setTo(uchar(0));
            midROI.setTo(uchar(0));
            detailROI.setTo(uchar(255));
            break;
        case BlockClass::None:
        default:
            // Leave all zeros, do not override anything.
            // (If you prefer to force exactly-one-class everywhere,
            //  you can choose a default here.)
            break;
        }
    }

    return out;
}

} // namespace regions
} // namespace iqa

