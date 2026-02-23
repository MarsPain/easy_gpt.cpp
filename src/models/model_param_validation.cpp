#include "models/model_param_validation.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "config.hpp"
#include "models/layer_key_prefix.hpp"
#include "models/loader.hpp"

namespace easy_llm {

namespace {

std::string shape_to_string(const std::vector<int>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string join_keys(const std::vector<std::string>& keys) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << keys[i];
    }
    return oss.str();
}

void expect_rank2_shape(const Tensor& tensor, const std::string& key) {
    std::vector<int> shape = tensor.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument("Invalid tensor rank for key '" + key + "': expected rank=2, got shape=" +
                                    shape_to_string(shape));
    }
}

void expect_matrix_shape(const Tensor& tensor, const std::string& key, int rows, int cols) {
    expect_rank2_shape(tensor, key);
    std::vector<int> shape = tensor.shape();
    if (shape[0] != rows || shape[1] != cols) {
        throw std::invalid_argument(
            "Invalid tensor shape for key '" + key + "': expected [" + std::to_string(rows) + ", " +
            std::to_string(cols) + "], got " + shape_to_string(shape));
    }
}

void expect_vector_shape(const Tensor& tensor, const std::string& key, int dim0) {
    std::vector<int> shape = tensor.shape();
    bool valid = false;
    if (shape.size() == 1) {
        valid = (shape[0] == dim0);
    } else if (shape.size() == 2) {
        valid = (shape[0] == dim0 && shape[1] == 1);
    }
    if (!valid) {
        throw std::invalid_argument("Invalid tensor shape for key '" + key + "': expected [" + std::to_string(dim0) +
                                    "] or [" + std::to_string(dim0) + ", 1], got " + shape_to_string(shape));
    }
}

void require_key(const ModelParam& model_param, const std::string& key, std::vector<std::string>& missing_keys) {
    if (!model_param.contains(key)) {
        missing_keys.push_back(key);
    }
}

} // namespace

void validate_model_params_before_load(const Config& config,
                                       const LayerKeyPrefix& key_prefix,
                                       const ModelParam& model_param) {
    if (config.num_layers <= 0) {
        throw std::invalid_argument("validate_model_params_before_load: num_layers must be > 0.");
    }
    if (config.hidden_size <= 0 || config.num_heads <= 0 || config.num_heads_kv <= 0) {
        throw std::invalid_argument(
            "validate_model_params_before_load: hidden_size/num_heads/num_heads_kv must be > 0.");
    }
    if (config.hidden_size % config.num_heads != 0) {
        throw std::invalid_argument(
            "validate_model_params_before_load: hidden_size must be divisible by num_heads.");
    }
    if (config.vocab_size <= 0) {
        throw std::invalid_argument("validate_model_params_before_load: vocab_size must be > 0.");
    }

    const int effective_head_dim =
        config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_heads);
    if (effective_head_dim <= 0) {
        throw std::invalid_argument("validate_model_params_before_load: effective head_dim must be > 0.");
    }
    const int q_hidden_size = config.num_heads * effective_head_dim;
    const int kv_hidden_size = config.num_heads_kv * effective_head_dim;

    const std::string embed_weight_key = "model.embed_tokens.weight";
    const std::string model_norm_weight_key = key_prefix.model_norm() + ".weight";

    std::vector<std::string> missing_keys;
    missing_keys.reserve(static_cast<std::size_t>(config.num_layers) * 8 + 2);
    require_key(model_param, embed_weight_key, missing_keys);
    require_key(model_param, model_norm_weight_key, missing_keys);

    struct LayerWeightKeys {
        std::string q_proj_weight;
        std::string k_proj_weight;
        std::string v_proj_weight;
        std::string o_proj_weight;
        std::string q_norm_weight;
        std::string k_norm_weight;
        std::string input_norm_weight;
        std::string down_proj_weight;
        std::string gate_proj_weight;
        std::string up_proj_weight;
        std::string post_attn_norm_weight;
    };

    std::vector<LayerWeightKeys> layer_keys;
    layer_keys.reserve(config.num_layers);
    for (int layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
        std::string layer_key = key_prefix.layer(layer_idx);
        LayerWeightKeys keys{
            key_prefix.self_attn_q_proj(layer_key) + ".weight",
            key_prefix.self_attn_k_proj(layer_key) + ".weight",
            key_prefix.self_attn_v_proj(layer_key) + ".weight",
            key_prefix.self_attn_o_proj(layer_key) + ".weight",
            key_prefix.self_attn_q_norm(layer_key) + ".weight",
            key_prefix.self_attn_k_norm(layer_key) + ".weight",
            key_prefix.input_layer_norm(layer_key) + ".weight",
            key_prefix.mlp_down_proj(layer_key) + ".weight",
            key_prefix.mlp_gate_proj(layer_key) + ".weight",
            key_prefix.mlp_up_proj(layer_key) + ".weight",
            key_prefix.post_attention_layer_norm(layer_key) + ".weight"
        };
        require_key(model_param, keys.q_proj_weight, missing_keys);
        require_key(model_param, keys.k_proj_weight, missing_keys);
        require_key(model_param, keys.v_proj_weight, missing_keys);
        require_key(model_param, keys.o_proj_weight, missing_keys);
        if (key_prefix.uses_qk_norm()) {
            require_key(model_param, keys.q_norm_weight, missing_keys);
            require_key(model_param, keys.k_norm_weight, missing_keys);
        }
        require_key(model_param, keys.input_norm_weight, missing_keys);
        require_key(model_param, keys.down_proj_weight, missing_keys);
        require_key(model_param, keys.gate_proj_weight, missing_keys);
        require_key(model_param, keys.up_proj_weight, missing_keys);
        require_key(model_param, keys.post_attn_norm_weight, missing_keys);
        layer_keys.push_back(std::move(keys));
    }

    if (!missing_keys.empty()) {
        throw std::invalid_argument("Missing required model weights: " + join_keys(missing_keys));
    }

    expect_matrix_shape(model_param.peek_param(embed_weight_key), embed_weight_key, config.vocab_size, config.hidden_size);
    expect_vector_shape(model_param.peek_param(model_norm_weight_key), model_norm_weight_key, config.hidden_size);

    for (const auto& keys : layer_keys) {
        expect_matrix_shape(model_param.peek_param(keys.q_proj_weight), keys.q_proj_weight, q_hidden_size, config.hidden_size);
        expect_matrix_shape(model_param.peek_param(keys.k_proj_weight), keys.k_proj_weight, kv_hidden_size, config.hidden_size);
        expect_matrix_shape(model_param.peek_param(keys.v_proj_weight), keys.v_proj_weight, kv_hidden_size, config.hidden_size);
        expect_matrix_shape(model_param.peek_param(keys.o_proj_weight), keys.o_proj_weight, config.hidden_size, q_hidden_size);
        if (key_prefix.uses_qk_norm()) {
            expect_vector_shape(model_param.peek_param(keys.q_norm_weight), keys.q_norm_weight, effective_head_dim);
            expect_vector_shape(model_param.peek_param(keys.k_norm_weight), keys.k_norm_weight, effective_head_dim);
        }
        expect_vector_shape(model_param.peek_param(keys.input_norm_weight), keys.input_norm_weight, config.hidden_size);
        expect_vector_shape(model_param.peek_param(keys.post_attn_norm_weight), keys.post_attn_norm_weight, config.hidden_size);

        const Tensor& down_proj = model_param.peek_param(keys.down_proj_weight);
        const Tensor& gate_proj = model_param.peek_param(keys.gate_proj_weight);
        const Tensor& up_proj = model_param.peek_param(keys.up_proj_weight);
        expect_rank2_shape(down_proj, keys.down_proj_weight);
        expect_rank2_shape(gate_proj, keys.gate_proj_weight);
        expect_rank2_shape(up_proj, keys.up_proj_weight);

        std::vector<int> down_shape = down_proj.shape();
        std::vector<int> gate_shape = gate_proj.shape();
        std::vector<int> up_shape = up_proj.shape();
        if (down_shape[0] != config.hidden_size) {
            throw std::invalid_argument("Invalid down_proj out dim for key '" + keys.down_proj_weight +
                                        "': expected " + std::to_string(config.hidden_size) + ", got " +
                                        std::to_string(down_shape[0]));
        }
        if (gate_shape[1] != config.hidden_size || up_shape[1] != config.hidden_size) {
            throw std::invalid_argument("Invalid gate/up proj input dim: expected hidden_size=" +
                                        std::to_string(config.hidden_size) + ", got gate=" +
                                        shape_to_string(gate_shape) + ", up=" + shape_to_string(up_shape));
        }
        if (gate_shape[0] != up_shape[0]) {
            throw std::invalid_argument("Invalid MLP intermediate dim mismatch: gate=" +
                                        std::to_string(gate_shape[0]) + ", up=" + std::to_string(up_shape[0]));
        }
        if (down_shape[1] != up_shape[0]) {
            throw std::invalid_argument("Invalid MLP down_proj input dim mismatch: down=" +
                                        std::to_string(down_shape[1]) + ", up_out=" + std::to_string(up_shape[0]));
        }
    }
}

void validate_no_remaining_model_params(const ModelParam& model_param) {
    std::vector<std::string> remaining = model_param.remaining_keys();
    if (remaining.empty()) {
        return;
    }

    constexpr std::size_t kPreviewLimit = 20;
    const std::size_t preview_size = std::min(remaining.size(), kPreviewLimit);
    std::vector<std::string> preview(remaining.begin(), remaining.begin() + preview_size);
    const std::string suffix =
        remaining.size() > kPreviewLimit
            ? " ... (+" + std::to_string(remaining.size() - kPreviewLimit) + " more)"
            : "";

    const std::string preview_text = join_keys(preview) + suffix;
    spdlog::warn(
        "Found {} unconsumed model weights after load (ignored): {}",
        remaining.size(),
        preview_text
    );
}

} // namespace easy_llm
