#pragma once

#include "artifact/reader.h"

#include <string_view>

namespace ninfer::artifact {

// v3 closed-registry spellings mapped onto this target's registered formats. Names that are
// registered upstream but excluded for this platform (nvfp4, fp8_e4m3fn_row_bf16, and their
// row_scale/block_scale layouts) are recognized and rejected with a target-specific error.
[[nodiscard]] NumericFormat parse_v3_format(std::string_view name);
[[nodiscard]] StorageLayout parse_v3_layout(std::string_view name);
[[nodiscard]] ResourceEncoding parse_v3_encoding(std::string_view name);

} // namespace ninfer::artifact