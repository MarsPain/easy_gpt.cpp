#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "config.hpp"
#include "models/layer_key_prefix.hpp"
#include "models/loader.hpp"
#include "models/model_param_validation.hpp"
#include "tensor.hpp"

namespace {

enum class FixtureDType {
    Bf16,
    F16,
    F32,
    I8
};

struct TensorSpec {
    std::string key;
    std::vector<int> shape;
    float fill_value;
    FixtureDType dtype{FixtureDType::Bf16};
};

std::size_t count_elements(const std::vector<int>& shape) {
    std::size_t total = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Fixture shape contains non-positive dimension.");
        }
        total *= static_cast<std::size_t>(dim);
    }
    return total;
}

const char* to_dtype_name(FixtureDType dtype) {
    switch (dtype) {
        case FixtureDType::Bf16: return "BF16";
        case FixtureDType::F16: return "F16";
        case FixtureDType::F32: return "F32";
        case FixtureDType::I8: return "I8";
        default: throw std::runtime_error("Unsupported fixture dtype.");
    }
}

std::size_t dtype_bytes(FixtureDType dtype) {
    switch (dtype) {
        case FixtureDType::Bf16: return sizeof(std::uint16_t);
        case FixtureDType::F16: return sizeof(std::uint16_t);
        case FixtureDType::F32: return sizeof(float);
        case FixtureDType::I8: return sizeof(std::int8_t);
        default: throw std::runtime_error("Unsupported fixture dtype.");
    }
}

std::uint16_t float_to_fp16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const int shift = 14 - exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        std::uint32_t round_bit = (mantissa >> (shift - 1)) & 1u;
        std::uint32_t sticky_bits = mantissa & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky_bits || (half_mantissa & 1u))) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }

    std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> 13);
    std::uint32_t round_bits = mantissa & 0x1FFFu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x0400u) {
            half_mantissa = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7C00u);
            }
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(exp) << 10) | half_mantissa);
}

template <typename T>
void append_scalar(std::vector<char>& bytes, T value) {
    const char* ptr = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), ptr, ptr + sizeof(T));
}

std::vector<char> build_tensor_blob(const TensorSpec& spec) {
    const std::size_t total = count_elements(spec.shape);
    std::vector<char> bytes;
    bytes.reserve(total * dtype_bytes(spec.dtype));

    for (std::size_t i = 0; i < total; ++i) {
        const float value = spec.fill_value + static_cast<float>(i) * 0.001f;
        switch (spec.dtype) {
            case FixtureDType::Bf16: {
                const std::uint16_t encoded = easy_llm::float_to_bf16(value);
                append_scalar(bytes, encoded);
                break;
            }
            case FixtureDType::F16: {
                const std::uint16_t encoded = float_to_fp16_bits(value);
                append_scalar(bytes, encoded);
                break;
            }
            case FixtureDType::F32: {
                append_scalar(bytes, value);
                break;
            }
            case FixtureDType::I8: {
                const std::int8_t encoded = static_cast<std::int8_t>(value);
                append_scalar(bytes, encoded);
                break;
            }
            default:
                throw std::runtime_error("Unsupported fixture dtype.");
        }
    }
    return bytes;
}

std::filesystem::path write_safetensors_fixture(const std::string& filename,
                                                const std::vector<TensorSpec>& specs) {
    nlohmann::json header = nlohmann::json::object();
    std::vector<char> binary_blob;
    std::uint64_t data_offset = 0;

    for (const auto& spec : specs) {
        std::vector<char> data = build_tensor_blob(spec);
        std::size_t bytes = data.size();

        std::uint64_t begin = data_offset;
        std::uint64_t end = data_offset + static_cast<std::uint64_t>(bytes);
        data_offset = end;

        header[spec.key] = {
            {"dtype", to_dtype_name(spec.dtype)},
            {"shape", spec.shape},
            {"data_offsets", {begin, end}}
        };

        const char* ptr = data.data();
        binary_blob.insert(binary_blob.end(), ptr, ptr + bytes);
    }

    std::string header_str = header.dump();
    std::uint64_t header_len = static_cast<std::uint64_t>(header_str.size());

    std::filesystem::path path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open fixture path: " + path.string());
    }

    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    if (!binary_blob.empty()) {
        out.write(binary_blob.data(), static_cast<std::streamsize>(binary_blob.size()));
    }
    out.close();
    return path;
}

easy_llm::Config make_test_config() {
    easy_llm::Config config;
    config.num_layers = 1;
    config.hidden_size = 4;
    config.num_heads = 2;
    config.num_heads_kv = 1;
    config.vocab_size = 10;
    config.architecture = "Qwen2ForCausalLM";
    config.model_type = "qwen2";
    return config;
}

easy_llm::Config make_qwen3_test_config() {
    easy_llm::Config config;
    config.num_layers = 1;
    config.hidden_size = 8;
    config.num_heads = 2;
    config.num_heads_kv = 1;
    config.head_dim = 6;
    config.vocab_size = 12;
    config.architecture = "Qwen3ForCausalLM";
    config.model_type = "qwen3";
    return config;
}

std::vector<TensorSpec> make_complete_specs() {
    return {
        {"model.embed_tokens.weight", {10, 4}, 1.0f},
        {"model.layers.0.self_attn.q_proj.weight", {4, 4}, 2.0f},
        {"model.layers.0.self_attn.k_proj.weight", {2, 4}, 3.0f},
        {"model.layers.0.self_attn.v_proj.weight", {2, 4}, 4.0f},
        {"model.layers.0.self_attn.o_proj.weight", {4, 4}, 5.0f},
        {"model.layers.0.input_layernorm.weight", {4}, 6.0f},
        {"model.layers.0.mlp.down_proj.weight", {4, 8}, 7.0f},
        {"model.layers.0.mlp.gate_proj.weight", {8, 4}, 8.0f},
        {"model.layers.0.mlp.up_proj.weight", {8, 4}, 9.0f},
        {"model.layers.0.post_attention_layernorm.weight", {4}, 10.0f},
        {"model.norm.weight", {4}, 11.0f}
    };
}

std::vector<TensorSpec> make_qwen3_specs() {
    return {
        {"model.embed_tokens.weight", {12, 8}, 1.0f},
        {"model.layers.0.self_attn.q_proj.weight", {12, 8}, 2.0f},
        {"model.layers.0.self_attn.k_proj.weight", {6, 8}, 3.0f},
        {"model.layers.0.self_attn.v_proj.weight", {6, 8}, 4.0f},
        {"model.layers.0.self_attn.o_proj.weight", {8, 12}, 5.0f},
        {"model.layers.0.self_attn.q_norm.weight", {6}, 5.5f},
        {"model.layers.0.self_attn.k_norm.weight", {6}, 5.8f},
        {"model.layers.0.input_layernorm.weight", {8}, 6.0f},
        {"model.layers.0.mlp.down_proj.weight", {8, 16}, 7.0f},
        {"model.layers.0.mlp.gate_proj.weight", {16, 8}, 8.0f},
        {"model.layers.0.mlp.up_proj.weight", {16, 8}, 9.0f},
        {"model.layers.0.post_attention_layernorm.weight", {8}, 10.0f},
        {"model.norm.weight", {8}, 11.0f}
    };
}

void consume_required_weights(const easy_llm::Config& config,
                              const easy_llm::LayerKeyPrefix& key_prefix,
                              easy_llm::ModelParam& model_param) {
    (void)model_param.take_param("model.embed_tokens.weight");
    for (int i = 0; i < config.num_layers; ++i) {
        const std::string layer_key = key_prefix.layer(i);
        (void)model_param.take_param(key_prefix.self_attn_q_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.self_attn_k_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.self_attn_v_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.self_attn_o_proj(layer_key) + ".weight");
        if (key_prefix.uses_qk_norm()) {
            (void)model_param.take_param(key_prefix.self_attn_q_norm(layer_key) + ".weight");
            (void)model_param.take_param(key_prefix.self_attn_k_norm(layer_key) + ".weight");
        }
        (void)model_param.take_param(key_prefix.input_layer_norm(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.mlp_down_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.mlp_gate_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.mlp_up_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.post_attention_layer_norm(layer_key) + ".weight");
    }
    (void)model_param.take_param(key_prefix.model_norm() + ".weight");
}

template <typename ExceptionType, typename Fn>
bool expect_throws(Fn&& fn, const char* case_name) {
    try {
        fn();
    } catch (const ExceptionType&) {
        return true;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << case_name << " threw wrong exception: " << e.what() << "\n";
        return false;
    }
    std::cerr << "FAIL: " << case_name << " should throw\n";
    return false;
}

bool run_take_and_peek_contract_case() {
    auto specs = make_complete_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_contract.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());

    const std::string key = "model.embed_tokens.weight";
    if (!model_param->contains(key)) {
        std::cerr << "FAIL: contains should be true before take\n";
        return false;
    }
    const auto& peeked = model_param->peek_param(key);
    if (peeked.shape() != std::vector<int>({10, 4})) {
        std::cerr << "FAIL: peek shape mismatch\n";
        return false;
    }
    auto taken = model_param->take_param(key);
    if (taken.shape() != std::vector<int>({10, 4})) {
        std::cerr << "FAIL: take shape mismatch\n";
        return false;
    }
    if (model_param->contains(key)) {
        std::cerr << "FAIL: contains should be false after take\n";
        return false;
    }
    if (!expect_throws<std::out_of_range>([&]() {
            (void)model_param->peek_param(key);
        }, "peek after take")) {
        return false;
    }
    return true;
}

bool run_loader_preserves_rank_case() {
    std::vector<TensorSpec> specs{
        {"model.embed_tokens.weight", {2, 3, 4}, 0.5f}
    };
    std::filesystem::path path = write_safetensors_fixture("easy_llm_loader_rank3.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());

    const auto& tensor = model_param->peek_param("model.embed_tokens.weight");
    if (tensor.shape() != std::vector<int>({2, 3, 4})) {
        std::cerr << "FAIL: loader should preserve original tensor rank\n";
        return false;
    }
    return true;
}

bool almost_equal(float lhs, float rhs, float abs_tol = 1e-2f) {
    return std::fabs(lhs - rhs) <= abs_tol;
}

bool run_loader_f32_conversion_case() {
    std::vector<TensorSpec> specs{
        {"model.embed_tokens.weight", {2, 2}, 1.25f, FixtureDType::F32}
    };
    std::filesystem::path path = write_safetensors_fixture("easy_llm_loader_f32.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    const auto& tensor = model_param->peek_param("model.embed_tokens.weight");
    if (tensor.shape() != std::vector<int>({2, 2})) {
        std::cerr << "FAIL: F32 conversion shape mismatch\n";
        return false;
    }
    if (!almost_equal(static_cast<float>(tensor.at(0)), 1.25f, 2e-2f)) {
        std::cerr << "FAIL: F32 conversion value mismatch at index 0\n";
        return false;
    }
    if (!almost_equal(static_cast<float>(tensor.at(3)), 1.253f, 2e-2f)) {
        std::cerr << "FAIL: F32 conversion value mismatch at index 3\n";
        return false;
    }
    return true;
}

bool run_loader_f16_conversion_case() {
    std::vector<TensorSpec> specs{
        {"model.embed_tokens.weight", {2, 2}, -0.75f, FixtureDType::F16}
    };
    std::filesystem::path path = write_safetensors_fixture("easy_llm_loader_f16.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    const auto& tensor = model_param->peek_param("model.embed_tokens.weight");
    if (tensor.shape() != std::vector<int>({2, 2})) {
        std::cerr << "FAIL: F16 conversion shape mismatch\n";
        return false;
    }
    if (!almost_equal(static_cast<float>(tensor.at(0)), -0.75f, 2e-2f)) {
        std::cerr << "FAIL: F16 conversion value mismatch at index 0\n";
        return false;
    }
    if (!almost_equal(static_cast<float>(tensor.at(3)), -0.747f, 2e-2f)) {
        std::cerr << "FAIL: F16 conversion value mismatch at index 3\n";
        return false;
    }
    return true;
}

bool run_loader_unsupported_dtype_case() {
    std::vector<TensorSpec> specs{
        {"model.embed_tokens.weight", {4, 1}, 2.0f, FixtureDType::I8}
    };
    std::filesystem::path path = write_safetensors_fixture("easy_llm_loader_i8.safetensors", specs);
    return expect_throws<std::runtime_error>([&]() {
        auto model_param = easy_llm::ModelParam::load(path.string());
        (void)model_param;
    }, "loader unsupported dtype");
}

bool run_validation_success_case() {
    auto specs = make_complete_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_valid.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    return true;
}

bool run_validation_missing_key_case() {
    auto specs = make_complete_specs();
    specs.erase(specs.begin() + 2);  // remove k_proj
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_missing.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    return expect_throws<std::invalid_argument>([&]() {
        easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    }, "validation missing key");
}

bool run_validation_bad_shape_case() {
    auto specs = make_complete_specs();
    specs[0].shape = {9, 4};  // embed vocab mismatch
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_bad_shape.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    return expect_throws<std::invalid_argument>([&]() {
        easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    }, "validation bad shape");
}

bool run_qwen3_validation_success_case() {
    auto specs = make_qwen3_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_qwen3_valid.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_qwen3_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    return true;
}

bool run_qwen3_validation_missing_qk_norm_case() {
    auto specs = make_qwen3_specs();
    specs.erase(specs.begin() + 5); // remove q_norm
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_qwen3_missing_qnorm.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_qwen3_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    return expect_throws<std::invalid_argument>([&]() {
        easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    }, "qwen3 validation missing q_norm");
}

bool run_qwen3_validation_bad_o_proj_shape_case() {
    auto specs = make_qwen3_specs();
    specs[4].shape = {8, 8}; // should be [hidden_size, num_heads * head_dim]
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_qwen3_bad_o_proj.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_qwen3_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    return expect_throws<std::invalid_argument>([&]() {
        easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    }, "qwen3 validation bad o_proj shape");
}

bool run_remaining_keys_success_case() {
    auto specs = make_complete_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_remaining_ok.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    consume_required_weights(config, *key_prefix, *model_param);
    easy_llm::validate_no_remaining_model_params(*model_param);
    if (model_param->size() != 0) {
        std::cerr << "FAIL: remaining key count should be 0\n";
        return false;
    }
    return true;
}

bool run_remaining_keys_fail_case() {
    auto specs = make_complete_specs();
    specs.push_back({"model.unused_adapter.weight", {2, 2}, 12.0f});
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_remaining_bad.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    consume_required_weights(config, *key_prefix, *model_param);
    easy_llm::validate_no_remaining_model_params(*model_param);
    if (model_param->size() != 1) {
        std::cerr << "FAIL: remaining key count should be 1\n";
        return false;
    }
    std::vector<std::string> remaining = model_param->remaining_keys();
    if (remaining.size() != 1 || remaining[0] != "model.unused_adapter.weight") {
        std::cerr << "FAIL: remaining key should be model.unused_adapter.weight\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!run_take_and_peek_contract_case()) {
        return 1;
    }
    if (!run_loader_preserves_rank_case()) {
        return 1;
    }
    if (!run_loader_f32_conversion_case()) {
        return 1;
    }
    if (!run_loader_f16_conversion_case()) {
        return 1;
    }
    if (!run_loader_unsupported_dtype_case()) {
        return 1;
    }
    if (!run_validation_success_case()) {
        return 1;
    }
    if (!run_validation_missing_key_case()) {
        return 1;
    }
    if (!run_validation_bad_shape_case()) {
        return 1;
    }
    if (!run_qwen3_validation_success_case()) {
        return 1;
    }
    if (!run_qwen3_validation_missing_qk_norm_case()) {
        return 1;
    }
    if (!run_qwen3_validation_bad_o_proj_shape_case()) {
        return 1;
    }
    if (!run_remaining_keys_success_case()) {
        return 1;
    }
    if (!run_remaining_keys_fail_case()) {
        return 1;
    }
    std::cout << "PASS: model param validation test\n";
    return 0;
}
