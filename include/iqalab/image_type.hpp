#pragma once

#include <filesystem>
#include <fstream>
namespace iqa {

enum class ImageType {
  Unknown = 0,
  Bmp,
  Jpeg,
  Png,
  Tiff,
  Pnm,   // PBM/PGM/PPM
  Webp,
  Jp2,   // JPEG 2000
  Gif,
  Avif
};

const char* to_string(ImageType t);

std::string lower_extension(const std::filesystem::path & p);

bool detect_image_type1(const std::filesystem::path& p);

ImageType get_image_type(const std::string& path);

bool is_image_file(const std::string& path);

} // namespace iqa
