#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer::test::artifact_fixture {

using Json = nlohmann::json;

inline constexpr std::array<std::uint8_t, 8> kV1Magic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 1,
};
inline constexpr std::array<std::uint8_t, 8> kMagic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 2,
};
inline constexpr std::array<std::uint8_t, 8> kV3Magic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 3,
};
inline constexpr std::array<std::uint8_t, 8> kV3PartMagic = {
    'N', 'I', 'N', 'P', 'R', 'T', 0, 3,
};
inline constexpr std::uint64_t kV3HeaderBytes      = 32;
inline constexpr std::uint64_t kV3PayloadAlignment = 4096;

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline void write_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { output[i] = std::byte((value >> (i * 8)) & 0xff); }
}

inline void write_magic(std::byte* output, const std::array<std::uint8_t, 8>& magic) {
    for (std::size_t i = 0; i < magic.size(); ++i) { output[i] = std::byte{magic[i]}; }
}

struct TemporaryArtifact {
    std::filesystem::path path;
    std::vector<std::filesystem::path> parts;

    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        for (const auto& part : parts) { std::filesystem::remove(part, ignored); }
    }
};

inline TemporaryArtifact write_fixture(const Json& directory, std::string_view suffix,
                                       const std::array<std::uint8_t, 8>& magic = kMagic) {
    const std::string json         = directory.dump();
    const auto payload_offset      = align_up(16 + json.size(), 4096);
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };
    std::uint64_t payload_bytes = 0;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            payload_bytes = std::max(payload_bytes, object.at("offset").get<std::uint64_t>() +
                                                        object.at("bytes").get<std::uint64_t>());
        }
    }

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < magic.size(); ++i) { file[i] = std::byte{magic[i]}; }
    write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + 16, json.data(), json.size());

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            const auto offset = object.at("offset").get<std::uint64_t>();
            const auto bytes  = object.at("bytes").get<std::uint64_t>();
            std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
        }
    }

    static std::atomic<std::uint64_t> counter{0};
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + "_" + std::to_string(pid) + "_" +
                 std::to_string(counter.fetch_add(1)) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write artifact fixture"); }
    return {std::move(path)};
}

inline TemporaryArtifact write_v3_fixture(const Json& directory, std::string_view suffix,
                                          std::array<std::byte, 16> artifact_id = {}) {
    const std::string json           = directory.dump();
    const auto entry_payload_offset  = align_up(kV3HeaderBytes + json.size(), kV3PayloadAlignment);
    const auto nonnegative_integer   = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };

    const auto& files = directory.at("files");
    std::vector<std::uint64_t> file_begin;
    std::vector<std::uint64_t> file_payload;
    std::uint64_t logical_begin = 0;
    for (const auto& record : files) {
        file_begin.push_back(logical_begin);
        file_payload.push_back(record.at("payload_bytes").get<std::uint64_t>());
        logical_begin += record.at("payload_bytes").get<std::uint64_t>();
    }

    // Entry file: 32-byte v3 header, JSON, padding, entry payload.
    auto entry = std::vector<std::byte>(entry_payload_offset + file_payload.front(), std::byte{0});
    write_magic(entry.data(), kV3Magic);
    write_u64_le(entry.data() + 8, json.size());
    for (std::size_t i = 0; i < artifact_id.size(); ++i) { entry[16 + i] = artifact_id[i]; }
    std::memcpy(entry.data() + kV3HeaderBytes, json.data(), json.size());

    // Continuation part files: 32-byte part header, padding to the 4096 payload slot.
    std::vector<std::vector<std::byte>> part_buffers;
    for (std::size_t part = 1; part < files.size(); ++part) {
        auto buffer = std::vector<std::byte>(kV3PayloadAlignment + file_payload[part], std::byte{0});
        write_magic(buffer.data(), kV3PartMagic);
        write_u64_le(buffer.data() + 8, part);
        for (std::size_t i = 0; i < artifact_id.size(); ++i) { buffer[16 + i] = artifact_id[i]; }
        part_buffers.push_back(std::move(buffer));
    }

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (!object.contains("offset") || !object.contains("bytes") ||
            !nonnegative_integer(object.at("offset")) ||
            !nonnegative_integer(object.at("bytes"))) {
            continue;
        }
        const auto offset = object.at("offset").get<std::uint64_t>();
        const auto bytes  = object.at("bytes").get<std::uint64_t>();
        for (std::size_t f = 0; f < file_begin.size(); ++f) {
            if (offset >= file_begin[f] && offset + bytes <= file_begin[f] + file_payload[f]) {
                const auto file_offset = offset - file_begin[f];
                if (f == 0) {
                    std::fill_n(entry.data() + entry_payload_offset + file_offset, bytes,
                                std::byte{marker});
                } else {
                    auto& buffer = part_buffers[f - 1];
                    std::fill_n(buffer.data() + kV3PayloadAlignment + file_offset, bytes,
                                std::byte{marker});
                }
                break;
            }
        }
        ++marker;
    }

    static std::atomic<std::uint64_t> v3_counter{0};
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    auto entry_path = std::filesystem::temp_directory_path() /
                      ("ninfer_v3_" + std::string(suffix) + "_" + std::to_string(pid) + "_" +
                       std::to_string(v3_counter.fetch_add(1)) + ".ninfer");
    std::ofstream output(entry_path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(entry.data()),
                 static_cast<std::streamsize>(entry.size()));
    if (!output) { throw std::runtime_error("failed to write v3 artifact fixture"); }

    TemporaryArtifact artifact{std::move(entry_path)};
    for (std::size_t part = 1; part < files.size(); ++part) {
        const auto& record = files[part];
        if (!record.at("path").is_string()) {
            throw std::runtime_error("continuation file record must name a sidecar path");
        }
        auto part_path = artifact.path.parent_path() / record.at("path").get<std::string>();
        std::ofstream part_output(part_path, std::ios::binary | std::ios::trunc);
        part_output.write(reinterpret_cast<const char*>(part_buffers[part - 1].data()),
                          static_cast<std::streamsize>(part_buffers[part - 1].size()));
        if (!part_output) { throw std::runtime_error("failed to write v3 part fixture"); }
        artifact.parts.push_back(std::move(part_path));
    }
    return artifact;
}

} // namespace ninfer::test::artifact_fixture