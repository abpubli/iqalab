#include "iqalab/utils/path_utils.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

namespace fs = std::filesystem;

namespace iqa::utils {

std::string lower_extension(const fs::path& p)
{
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return ext;
}

std::string stem_lower(const fs::path& p)
{
  std::string stem = p.stem().string();
  std::transform(stem.begin(), stem.end(), stem.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return stem;
}

} // namespace iqa::utils

