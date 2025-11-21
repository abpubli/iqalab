#include "iqalab/utils/file_grouping.hpp"

#include "iqalab/image_type.hpp"
#include "iqalab/utils/path_utils.hpp"

#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace iqa {
namespace utils {

static std::string to_lower_copy(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return out;
}

static bool icase_less(const std::filesystem::path& a,
                       const std::filesystem::path& b)
{
    return to_lower_copy(a.filename().string()) <
           to_lower_copy(b.filename().string());
}

std::vector<fs::path>
collect_reference_files(const fs::path& refDir)
{
    std::vector<fs::path> refs;
    if (!fs::exists(refDir) || !fs::is_directory(refDir)) {
        return refs;
    }

    for (auto& entry : fs::directory_iterator(refDir)) {
        const fs::path& p = entry.path();
        if (is_image_file(p)) {
            refs.push_back(p);
        }
    }

    std::sort(refs.begin(), refs.end(), icase_less);
    return refs;
}

std::vector<fs::path>
collect_distorted_files(const fs::path& distDir)
{
    std::vector<fs::path> dists;
    if (!fs::exists(distDir) || !fs::is_directory(distDir)) {
        return dists;
    }

    for (auto& entry : fs::directory_iterator(distDir)) {
        const fs::path& p = entry.path();
        if (is_image_file(p)) {
            dists.push_back(p);
        }
    }

    std::sort(dists.begin(), dists.end(), icase_less);
    return dists;
}

FileGroups group_distorted_by_reference(
    const std::vector<fs::path>& refFiles,
    const std::vector<fs::path>& distFiles
)
{
    FileGroups groups;

    // Precompute lowercase stems for distorted files
    struct DistInfo {
        std::string stemLower;
        fs::path path;
    };

    std::vector<DistInfo> distInfos;
    distInfos.reserve(distFiles.size());
    for (const auto& dist : distFiles) {
        DistInfo info;
        info.stemLower = stem_lower(dist);
        info.path      = dist;
        distInfos.push_back(std::move(info));
    }

    for (const auto& ref : refFiles) {
        std::string refStemLower = stem_lower(ref);
        if (refStemLower.empty()) {
            continue;
        }

        auto& vec = groups[refStemLower];  // creates vector if missing

        for (const auto& dinfo : distInfos) {
            // starts_with(refStemLower)
            if (dinfo.stemLower.rfind(refStemLower, 0) == 0) {
                vec.push_back(dinfo.path);
            }
        }

        // keep deterministic ordering
        std::sort(vec.begin(), vec.end(), icase_less);
    }

    return groups;
}

} // namespace utils
} // namespace iqa
