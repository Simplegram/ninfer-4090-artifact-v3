#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;
using ninfer::test::artifact_fixture::write_v3_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows   = StorageLayout::RowSplitK128V1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 8 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 8> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.file_index != 0 ||
            payload.file_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        auto normative = normative_directory();
        nlohmann::json directory{
            {"model_id", "qwen3.6-27b"},
            {"objects", normative.at("objects")},
        };
        auto fixture =
            write_fixture(directory, "legacy_v1", ninfer::test::artifact_fixture::kV1Magic);
        Reader reader(fixture.path);
        if (reader.identity().model_id != "qwen3.6-27b" ||
            reader.identity().weights_id != "groupwise-int") {
            throw std::runtime_error("legacy v1 identity mapping mismatch");
        }
    }
}

Json v3_directory(const std::vector<std::uint64_t>& logical_spans, Json objects,
                  std::string model_id = "fixture-model",
                  std::string weights_id = "fixture-weights") {
    Json files = Json::array();
    for (std::size_t i = 0; i < logical_spans.size(); ++i) {
        files.push_back(
            i == 0
                ? Json{{"path", nullptr}, {"payload_bytes", logical_spans[i]}}
                : Json{{"path", "part-" + std::to_string(i - 1) + ".ninfer"},
                       {"payload_bytes", logical_spans[i]}});
    }
    return {
        {"components", {{"text", {{"config", Json::object()}, {"target", "text"}}}}},
        {"objects", std::move(objects)},
        {"bindings", Json::object()},
        {"uses", Json::array()},
        {"files", std::move(files)},
        {"metadata", {{"name", model_id}, {"weights_id", weights_id}}},
        {"provenance", Json::object()},
    };
}

std::array<std::byte, 16> v3_artifact_id() {
    std::array<std::byte, 16> id{};
    for (std::size_t i = 0; i < id.size(); ++i) { id[i] = static_cast<std::byte>(i + 1); }
    return id;
}

Json v3_objects_from(const Json& normative_objects) {
    Json objects = Json::array();
    for (const auto& object : normative_objects) {
        Json item;
        item["id"]     = object.at("name");
        item["kind"]   = object.at("kind");
        if (object.at("kind") == "tensor") {
            item["shape"]  = object.at("shape");
            item["format"] = object.at("format");
            item["layout"] = object.at("layout");
        } else {
            item["encoding"] = object.at("encoding");
        }
        item["offset"] = object.at("offset");
        item["bytes"]  = object.at("bytes");
        objects.push_back(std::move(item));
    }

    return objects;
}

void test_v3_fixture() {
    auto objects = v3_objects_from(normative_directory().at("objects"));
    auto fixture = write_v3_fixture(v3_directory({4096}, std::move(objects)), "v3_valid",
                                    v3_artifact_id());
    Reader reader(fixture.path);
    if (!reader.is_v3()) {
        throw std::runtime_error("v3 fixture was not recognized as v3");
    }
    if (reader.artifact_id() != v3_artifact_id()) {
        throw std::runtime_error("artifact id did not survive the round trip");
    }
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 8) {
        throw std::runtime_error("v3 fixture identity mismatch");
    }

    for (std::size_t i = 0; i < reader.objects().size(); ++i) {
        const auto& object = reader.objects()[i];
        if (reader.find(ninfer::artifact::object_name(object)) != &object) {
            throw std::runtime_error("v3 name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.file_index != 0 ||
            payload.file_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("v3 payload span mismatch");
        }
    }

    const auto* q5 = std::get_if<TensorDescriptor>(reader.find("q5"));
    if (q5 == nullptr || q5->format != NumericFormat::Q5G64_F16S ||
        q5->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("v3 tensor signature mismatch");
    }
}

void test_v3_continuation() {
    Json objects = Json::array({
        {{"id", "entry/tensor"},
         {"kind", "tensor"},
         {"shape", {2}},
         {"format", "bf16"},
         {"layout", "contiguous_le_v1"},
         {"offset", 256},
         {"bytes", 4}},
        {{"id", "part/tensor"},
         {"kind", "tensor"},
         {"shape", {2}},
         {"format", "bf16"},
         {"layout", "contiguous_le_v1"},
         {"offset", 4096 + 256},
         {"bytes", 4}},
    });
    auto fixture =
        write_v3_fixture(v3_directory({4096, 4096}, std::move(objects)), "v3_continuation",
                         v3_artifact_id());
    Reader reader(fixture.path);
    if (fixture.parts.size() != 1) {
        throw std::runtime_error("continuation part file was not written");
    }

    const auto part = reader.find("part/tensor");
    if (part == nullptr || std::get_if<TensorDescriptor>(part) == nullptr) {
        throw std::runtime_error("part tensor missing or mistyped");
    }
    const auto part_payload = reader.payload(*part);
    if (part_payload.file_index != 1 || part_payload.file_offset != 4096 + 256 ||
        part_payload.data.front() != std::byte{2} || part_payload.data.back() != std::byte{2}) {
        throw std::runtime_error("part payload span mismatch");
    }
    const auto entry_payload = reader.payload(*reader.find("entry/tensor"));
    if (entry_payload.file_index != 0 || entry_payload.file_offset != 4096 + 256 ||
        entry_payload.data.front() != std::byte{1}) {
        throw std::runtime_error("entry payload span mismatch in a multi-file artifact");
    }
    if (reader.file_bytes() != 4 * 4096) {
        throw std::runtime_error("declared file length is wrong for a two-file artifact");
    }
}

void test_v3_rejections() {
    {
        auto directory = v3_directory({4096}, Json::array());
        directory["objects"] = Json::array({
            {{"id", "t"},
             {"kind", "tensor"},
             {"shape", {1}},
             {"format", "bf16"},
             {"layout", "contiguous_le_v1"},
             {"offset", 256},
             {"bytes", 2}},
        });
        directory["extra_member"] = true;
        auto fixture = write_v3_fixture(std::move(directory), "v3_extra_member");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "extra member");
    }
    {
        auto fixture = write_v3_fixture(
            v3_directory({4096}, Json::array({
                {{"id", "t"},
                 {"kind", "tensor"},
                 {"shape", {1}},
                 {"format", "bf16"},
                 {"layout", "contiguous_le_v1"},
                 {"offset", 256},
                 {"bytes", 3}},
            })),
            "v3_bad_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "v3 encoded size");
    }
    {
        auto fixture = write_v3_fixture(
            v3_directory({4096}, Json::array({
                {{"id", "t"},
                 {"kind", "tensor"},
                 {"shape", {1}},
                 {"format", "bf16"},
                 {"layout", "contiguous_le_v1"},
                 {"offset", 4096},
                 {"bytes", 2}},
            })),
            "v3_beyond_payload");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "beyond payload");
    }
    {
        auto directory = v3_directory(
            {4096},
            Json::array({
                {{"id", "t"},
                 {"kind", "tensor"},
                 {"shape", {1}},
                 {"format", "bf16"},
                 {"layout", "contiguous_le_v1"},
                 {"offset", 256},
                 {"bytes", 2}},
            }));
        directory.at("metadata").erase("name");
        auto fixture = write_v3_fixture(std::move(directory), "v3_no_name");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "metadata name");
    }
    {
        auto directory = v3_directory(
            {4096},
            Json::array({
                {{"id", "t"},
                 {"kind", "tensor"},
                 {"shape", {1}},
                 {"format", "nvfp4"},
                 {"layout", "contiguous_le_v1"},
                 {"offset", 256},
                 {"bytes", 2}},
            }));
        auto fixture = write_v3_fixture(std::move(directory), "v3_unsupported_format");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "unsupported format");
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        test_v3_fixture();
        test_v3_continuation();
        test_v3_rejections();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}