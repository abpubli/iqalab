#pragma once

#include <string>
#include <filesystem>

namespace iqa::utils {

// Return lowercase extension, e.g. ".jpg", ".png", ".bmp".
// If no extension: empty string.
std::string lower_extension(const std::filesystem::path& p);

// Return lowercase stem (filename without extension).
// E.g. "I01.BMP" -> "i01".
std::string stem_lower(const std::filesystem::path& p);

} // namespace iqa::utils

