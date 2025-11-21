#include "iqalab/image_type.hpp"

#include "iqalab/utils/path_utils.hpp"

#include <algorithm>
#include <cstring>

const char *iqa::to_string(ImageType t) {
  switch (t) {
  case ImageType::Bmp:
    return "bmp";
  case ImageType::Jpeg:
    return "jpeg";
  case ImageType::Png:
    return "png";
  case ImageType::Tiff:
    return "tiff";
  case ImageType::Pnm:
    return "pnm";
  case ImageType::Webp:
    return "webp";
  case ImageType::Jp2:
    return "jp2";
  case ImageType::Gif:
    return "gif";
  case ImageType::Avif:
    return "avif";
  case ImageType::Unknown:
  default:
    return "unknown";
  }
}

bool iqa::detect_image_type(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return false;

  unsigned char header[12] = {0};
  f.read(reinterpret_cast<char *>(header), sizeof(header));

  // JPEG
  if (header[0] == 0xFF && header[1] == 0xD8)
    return true;

  // PNG
  if (header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E &&
      header[3] == 0x47 && header[4] == 0x0D && header[5] == 0x0A &&
      header[6] == 0x1A && header[7] == 0x0A)
    return true;

  // BMP
  if (header[0] == 'B' && header[1] == 'M')
    return true;

  // GIF
  if (std::memcmp(header, "GIF87a", 6) == 0 ||
      std::memcmp(header, "GIF89a", 6) == 0)
    return true;

  // WebP: RIFF....WEBP
  if (std::memcmp(header, "RIFF", 4) == 0 &&
      std::memcmp(header + 8, "WEBP", 4) == 0)
    return true;

  // TIFF
  if ((header[0] == 0x49 && header[1] == 0x49 && header[2] == 0x2A &&
       header[3] == 0x00) ||
      (header[0] == 0x4D && header[1] == 0x4D && header[2] == 0x00 &&
       header[3] == 0x2A))
    return true;

  // AVIF: "ftypavif" (offset 4)
  if (std::memcmp(header + 4, "avif", 4) == 0 ||
      std::memcmp(header + 4, "heic", 4) == 0)
    return true;

  return false;
}

iqa::ImageType iqa::get_image_type(const std::string &path) {
  namespace fs = std::filesystem;

  fs::path p(path);
  const std::string ext = utils::lower_extension(p);

  if (ext == ".bmp" || ext == ".dib") {
    return ImageType::Bmp;
  }
  if (ext == ".jpg" || ext == ".jpeg" || ext == ".jpe") {
    return ImageType::Jpeg;
  }
  if (ext == ".png") {
    return ImageType::Png;
  }
  if (ext == ".tif" || ext == ".tiff") {
    return ImageType::Tiff;
  }
  if (ext == ".pbm" || ext == ".pgm" || ext == ".ppm" || ext == ".pnm") {
    return ImageType::Pnm;
  }
  if (ext == ".webp") {
    return ImageType::Webp;
  }
  if (ext == ".jp2" || ext == ".j2k" || ext == ".j2c") {
    return ImageType::Jp2;
  }
  if (ext == ".gif") {
    return ImageType::Gif;
  }
  if (ext == ".avif") {
    return ImageType::Avif;
  }
  return ImageType::Unknown;
}

bool iqa::is_image_file(const std::string &path) {
  // Note: this is a “declarative” list of types that are typically
  // supported by OpenCV – actual support depends
  // on which libraries it has been linked with.
  switch (get_image_type(path)) {
  case ImageType::Bmp:
  case ImageType::Jpeg:
  case ImageType::Png:
  case ImageType::Tiff:
  case ImageType::Pnm:
  case ImageType::Webp:
  case ImageType::Jp2:
  case ImageType::Gif:
    // Avif is optional – add if needed:
    // case ImageType::Avif:
    return true;

  case ImageType::Avif:
    // If you know you have OpenCV with libavif, you can put `return true;` here.
    return false;

  case ImageType::Unknown:
  default:
    return false;
  }
}