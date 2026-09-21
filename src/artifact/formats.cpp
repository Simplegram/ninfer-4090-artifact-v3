#include "artifact/formats.h"

#include <array>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{NumericFormat::BF16, std::string_view{"bf16"}},
    std::pair{NumericFormat::FP32, std::string_view{"fp32"}},
    std::pair{NumericFormat::I32, std::string_view{"int32"}},
    std::pair{NumericFormat::Q4G64_F16S, std::string_view{"q4_g64_fp16"}},
    std::pair{NumericFormat::Q5G64_F16S, std::string_view{"q5_g64_fp16"}},
    std::pair{NumericFormat::Q6G64_F16S, std::string_view{"q6_g64_fp16"}},
    std::pair{NumericFormat::W8G32_F16S, std::string_view{"q8_g32_fp16"}},
};
constexpr std::array kUnsupportedFormats = {
    std::string_view{"nvfp4"},
    std::string_view{"fp8_e4m3fn_row_bf16"},
};
constexpr std::array kLayouts = {
    std::pair{StorageLayout::ContiguousLeV1, std::string_view{"contiguous_le_v1"}},
    std::pair{StorageLayout::RowSplitK128V1, std::string_view{"row_split_k128_v1"}},
};
constexpr std::array kUnsupportedLayouts = {
    std::string_view{"row_scale_v1"},
    std::string_view{"block_scale_k16_m128x4_v1"},
};
constexpr std::array kEncodings = {
    std::pair{ResourceEncoding::RawBytesV1, std::string_view{"raw_bytes_v1"}},
};

} // namespace

NumericFormat parse_v3_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    for (const auto& spelling : kUnsupportedFormats) {
        if (spelling == name) {
            throw ArtifactError("tensor format '" + std::string(name) +
                                "' is not supported on this target");
        }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_v3_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    for (const auto& spelling : kUnsupportedLayouts) {
        if (spelling == name) {
            throw ArtifactError("tensor layout '" + std::string(name) +
                                "' is not supported on this target");
        }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_v3_encoding(std::string_view name) {
    for (const auto& [encoding, spelling] : kEncodings) {
        if (spelling == name) { return encoding; }
    }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

} // namespace ninfer::artifact