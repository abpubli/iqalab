#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>

namespace iqa {
namespace utils {

using FileGroups = std::unordered_map<std::string, std::vector<std::filesystem::path>>;

// Collect all supported image files from directory (non-recursive),
// sorted lexicographically by path.
std::vector<std::filesystem::path>
collect_reference_files(const std::filesystem::path& refDir);

// Collect all supported distorted image files from directory (non-recursive),
// sorted lexicographically by path.
std::vector<std::filesystem::path>
collect_distorted_files(const std::filesystem::path& distDir);

// Group distorted files by reference basename (TID-like convention).
//
// For each reference file:
//   refStemLower = stem_lower(ref)  e.g. "I01.BMP" -> "i01"
//
// For each distorted file:
//   distStemLower = stem_lower(dist)  e.g. "i01_01_1.bmp" -> "i01_01_1"
//
// A distorted file belongs to a reference group if:
//   distStemLower starts with refStemLower
// i.e. distStemLower.rfind(refStemLower, 0) == 0
//
// The resulting map is:
//   key = refStemLower
//   value = vector<dist_paths>, sorted lexicographically.
//
// This is suitable for TID-like datasets with naming:
//   I01.BMP  ->  i01_01_1.bmp, i01_01_2.bmp, ...
//
FileGroups group_distorted_by_reference(
    const std::vector<std::filesystem::path>& refFiles,
    const std::vector<std::filesystem::path>& distFiles
);

} // namespace utils
} // namespace iqa
