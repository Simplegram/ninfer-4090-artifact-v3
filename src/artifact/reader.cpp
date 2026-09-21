#include "artifact/reader.h"

#include "artifact/formats.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::array<std::byte, 8> kV2Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::uint64_t kPrefixBytes = 16;


template <std::size_t N>
void require_v2_members(const Json& value, const std::array<const char*, N>& members,
                        std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_v2_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_v2_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_v2_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_v2_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_v2_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_v2_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_v2_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_v2_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_v2_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_v2_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_v2_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_v2_tensor(value); }
    if (kind == "resource") { return parse_v2_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        mapping_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (mapping_file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(mapping_file_, &file_size)) {
            const auto error = ::GetLastError();
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError("artifact size does not fit the process address space");
        }

        size_ = static_cast<std::size_t>(file_size.QuadPart);
        if (size_ != 0) {
            mapping_ = ::CreateFileMappingW(mapping_file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_file_);
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            const void* view = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
            if (view == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_);
                ::CloseHandle(mapping_file_);
                mapping_      = nullptr;
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
            data_ = static_cast<const std::byte*>(view);
        }

        direct_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                         FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                     nullptr);
        if (direct_file_ == INVALID_HANDLE_VALUE) {
            const auto error = ::GetLastError();
            if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
            if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
            ::CloseHandle(mapping_file_);
            data_         = nullptr;
            mapping_      = nullptr;
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "CreateFileW direct " + path.string());
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (direct_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_file_); }
        if (mapping_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(mapping_file_); }
#else
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not alignment compliant");
        }
#ifdef _WIN32
        std::size_t total = 0;
        while (total < destination.size()) {
            const std::size_t amount =
                std::min<std::size_t>(destination.size() - total, 4 * 1024 * 1024);
            OVERLAPPED operation {};
            operation.Offset =
                static_cast<DWORD>(std::min<std::uint64_t>(absolute_offset + total, 0xFFFFFFFF));
            operation.OffsetHigh = static_cast<DWORD>((absolute_offset + total) >> 32);
            DWORD bytes = 0;
            const BOOL started = ::ReadFile(direct_file_, destination.data() + total, amount,
                                            &bytes, &operation);
            if (!started) {
                const auto error = ::GetLastError();
                if (error != ERROR_IO_PENDING ||
                    !::GetOverlappedResult(direct_file_, &operation, &bytes, TRUE)) {
                    const auto final_error = error;
                    throw std::system_error(static_cast<int>(final_error), std::system_category(),
                                            "direct artifact read");
                }
            }
            total += bytes;
            if (bytes != amount) { break; }
        }
        return total;
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
#ifdef _WIN32
    HANDLE mapping_file_       = INVALID_HANDLE_VALUE;
    HANDLE direct_file_        = INVALID_HANDLE_VALUE;
    HANDLE mapping_            = nullptr;
#else
    int fd_                = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    std::filesystem::path entry;
    bool v3                 = false;
    Directory directory;
    ArtifactId id           = {};
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start       = 0;
    std::uint64_t declared_file_bytes = 0;
    mutable std::vector<std::unique_ptr<MappedFile>> files;

    explicit Impl(const std::filesystem::path& path) : entry(path) {
        auto file = std::make_unique<MappedFile>(path);
        const auto size = file->size();
        if (size >= kHeaderBytes) {
            const auto* head = file->data();
            if (std::equal(kEntryMagic.begin(), kEntryMagic.end(), head)) {
                load_v3(std::move(file), head, path);
                return;
            }
        }
        if (size >= kPrefixBytes) {
            const auto* head = file->data();
            const bool legacy_v1 = std::equal(kV1Magic.begin(), kV1Magic.end(), head);
            if (!legacy_v1 && !std::equal(kV2Magic.begin(), kV2Magic.end(), head)) {
                throw ArtifactError("artifact magic is not NInfer v1, v2, or v3");
            }
            load_legacy(std::move(file), legacy_v1);
            return;
        }
        throw ArtifactError("artifact is shorter than the v2 prefix");
    }

    MappedFile& file(std::size_t index) const {
        if (index >= files.size()) { throw ArtifactError("invalid continuation index"); }
        if (!files[index]) {
            const auto& record = directory.files[index];
            auto input         = std::make_unique<MappedFile>(entry.parent_path() / *record.path);
            if (input->size() < kHeaderBytes) {
                throw ArtifactError(*record.path + ": continuation is shorter than the v3 header");
            }
            const auto* head = input->data();
            if (!std::equal(kPartMagic.begin(), kPartMagic.end(), head) ||
                read_u64_le(head + 8) != index || !std::equal(id.begin(), id.end(), head + 16)) {
                throw ArtifactError(*record.path +
                                    ": continuation header does not belong to this entry");
            }
            if (input->size() != checked_add(kPayloadAlignment, record.payload_bytes,
                                             "part length")) {
                throw ArtifactError(*record.path + ": continuation length differs from directory");
            }
            files[index] = std::move(input);
        }
        return *files[index];
    }

    void validate_entry(ObjectDescriptor entry, std::uint64_t payload_bytes) {
        const std::string name(object_name(entry));
        const auto offset     = object_offset(entry);
        const auto bytes      = object_bytes(entry);
        const auto alignment = std::visit(
            [](const auto& descriptor) {
                using Descriptor = std::decay_t<decltype(descriptor)>;
                if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                    return tensor_alignment(descriptor.layout);
                } else {
                    return resource_alignment(descriptor.encoding);
                }
            },
            entry);

        if (offset < validated_cursor) {
            throw ArtifactError(std::string("object ") + name + " overlaps or is out of order");
        }
        if (offset % alignment != 0) {
            throw ArtifactError(std::string("object ") + name + " is not " +
                                std::to_string(alignment) + "-byte aligned");
        }
        const auto end = checked_add(offset, bytes, "object payload range");
        if (end > payload_bytes) {
            throw ArtifactError(std::string("object ") + name + " extends beyond the payload");
        }
        const auto object_index = entries.size();
        auto [_, inserted]      = index.emplace(name, object_index);
        if (!inserted) {
            throw ArtifactError(std::string("duplicate object name: ") + name);
        }
        entries.push_back(std::move(entry));
        validated_cursor = end;
    }

    void load_v3(std::unique_ptr<MappedFile> entry_file, const std::byte* head,
                 const std::filesystem::path& path) {
        v3 = true;
        const auto json_bytes = read_u64_le(head + 8);
        if (!json_bytes || json_bytes > entry_file->size() - kHeaderBytes ||
            json_bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("artifact directory length is invalid");
        }
        std::copy_n(head + 16, id.size(), id.begin());

        std::string text(static_cast<std::size_t>(json_bytes), '\0');
        std::copy_n(reinterpret_cast<const char*>(head + kHeaderBytes),
                    static_cast<std::size_t>(json_bytes), text.data());
        directory = parse_directory(parse_json(text, "artifact directory"),
                                    path.filename().string());

        payload_start =
            align_up(checked_add(kHeaderBytes, json_bytes, "metadata end"), kPayloadAlignment,
                     "entry payload start");
        const auto expected =
            checked_add(payload_start, directory.files[0].payload_bytes, "entry length");
        if (expected != entry_file->size()) {
            throw ArtifactError("entry length differs from the artifact directory");
        }
        declared_file_bytes = checked_add(payload_start, directory.payload_bytes, "file bytes");
        declared_file_bytes = checked_add(
            declared_file_bytes,
            checked_mul(directory.files.size() - 1, kPayloadAlignment, "continuation headers"),
            "file bytes");

        files.push_back(std::move(entry_file));
        files.resize(directory.files.size());

        entries.reserve(directory.objects.size());
        index.reserve(directory.objects.size());
        for (const auto& object : directory.objects) {
            ObjectDescriptor entry;
            if (const auto* tensor = std::get_if<TensorObject>(&object)) {
                auto descriptor = TensorDescriptor{
                    tensor->id,
                    tensor->shape,
                    parse_v3_format(tensor->format),
                    parse_v3_layout(tensor->layout),
                    tensor->offset,
                    tensor->bytes,
                };
                const auto expected = tensor_encoded_size(
                    descriptor.layout, descriptor.format,
                    {descriptor.shape.data(), descriptor.shape.size()});
                if (descriptor.bytes != expected) {
                    throw ArtifactError("tensor " + descriptor.name + " stores " +
                                        std::to_string(descriptor.bytes) +
                                        " bytes; layout requires " + std::to_string(expected));
                }
                entry = std::move(descriptor);
            } else {
                const auto& resource = std::get<ResourceObject>(object);
                entry                = ResourceDescriptor{
                    resource.id,
                    parse_v3_encoding(resource.encoding),
                    resource.offset,
                    resource.bytes,
                };
            }
            validate_entry(std::move(entry), directory.payload_bytes);
        }
        for (const auto& object : directory.objects) {
            const auto offset = object_offset(object);
            const auto end    = checked_add(offset, object_bytes(object), "object payload range");
            bool inside       = false;
            for (const auto& record : directory.files) {
                if (offset >= record.logical_begin &&
                    end <= checked_add(record.logical_begin, record.payload_bytes,
                                       "file payload range")) {
                    inside = true;
                    break;
                }
            }
            if (!inside) {
                throw ArtifactError("object " + object_id(object) +
                                    " crosses continuation file boundaries");
            }
        }

        if (!directory.metadata.contains("name") ||
            !directory.metadata.contains("weights_id")) {
            throw ArtifactError("artifact metadata is missing name or weights_id");
        }
        identity.model_id   = require_id(directory.metadata.at("name"), "metadata name");
        identity.weights_id = require_id(directory.metadata.at("weights_id"), "metadata weights_id");
    }

    void load_legacy(std::unique_ptr<MappedFile> entry_file, bool legacy_v1) {
        const auto json_bytes = read_u64_le(entry_file->data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > entry_file->size() || payload_start > entry_file->size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(entry_file->data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        if (legacy_v1) {
            static constexpr std::array root_members = {"model_id", "objects"};
            require_v2_members(directory, root_members, "legacy directory root");
            identity.model_id = require_string(directory.at("model_id"), "model_id");
            if (identity.model_id != "qwen3.6-27b" && identity.model_id != "qwen3.6-35b-a3b") {
                throw ArtifactError("legacy artifact model_id is not a registered groupwise target");
            }
            identity.weights_id = "groupwise-int";
        } else {
            static constexpr std::array root_members = {"identity", "objects"};
            require_v2_members(directory, root_members, "directory root");
            const auto& raw_identity = directory.at("identity");
            static constexpr std::array identity_members = {"model_id", "weights_id"};
            require_v2_members(raw_identity, identity_members, "artifact identity");
            identity.model_id = require_string(raw_identity.at("model_id"), "model_id");
            identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");
        }

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = entry_file->size() - payload_start;
        validated_cursor         = 0;
        for (const auto& raw_object : raw_objects) {
            auto object = parse_v2_object(raw_object);
            validate_entry(std::move(object), payload_bytes);
        }

        files.push_back(std::move(entry_file));
    }

    std::uint64_t validated_cursor = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

bool Reader::is_v3() const noexcept { return impl_->v3; }

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const Directory& Reader::directory() const noexcept { return impl_->directory; }

const ArtifactId& Reader::artifact_id() const noexcept { return impl_->id; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto found = impl_->index.find(name);
    if (found == impl_->index.end()) { return nullptr; }
    return &impl_->entries[found->second];
}

std::uint64_t Reader::file_bytes() const noexcept {
    return impl_->v3 ? impl_->declared_file_bytes : impl_->file(0).size();
}

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const std::string name(object_name(object));
    const auto offset = object_offset(object);
    const auto bytes  = object_bytes(object);
    if (impl_->v3) {
        for (std::size_t index = 0; index < impl_->directory.files.size(); ++index) {
            const auto& record = impl_->directory.files[index];
            if (offset < record.logical_begin) { continue; }
            if (offset >= checked_add(record.logical_begin, record.payload_bytes,
                                      "file payload range")) {
                continue;
            }
            const auto physical =
                (index == 0 ? impl_->payload_start : kPayloadAlignment) +
                (offset - record.logical_begin);
            auto& input = impl_->file(index);
            if (physical + bytes > input.size()) {
                throw ArtifactError(std::string("object ") + name + " extends beyond its file");
            }
            return PayloadSpan{index, physical, {input.data() + physical, bytes}};
        }
        throw ArtifactError("object is outside every recorded file");
    }
    const auto physical = impl_->payload_start + offset;
    auto& input         = impl_->file(0);
    if (physical + bytes > input.size()) {
        throw ArtifactError(std::string("object ") + name + " extends beyond its file");
    }
    return PayloadSpan{0, physical, {input.data() + physical, bytes}};
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto object = find(name);
    if (object == nullptr) {
        throw ArtifactError("artifact object is missing: " + std::string(name));
    }
    return payload(*object);
}

std::size_t Reader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                std::span<std::byte> destination) const {
    return impl_->file(file_index).read_direct(file_offset, destination);
}

} // namespace ninfer::artifact