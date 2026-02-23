#include "config.hpp"

#include "spdlog/spdlog.h"

#include "utils.hpp"

namespace easy_llm {

void Config::load_config() {
    auto json_data = utils::load_json(model_config_path);
    if (json_data.empty()) {
        spdlog::error("Config load failed: {} is empty or missing", model_config_path);
        return;
    }

    num_layers = json_data.value("num_hidden_layers", num_layers);
    num_heads = json_data.value("num_attention_heads", num_heads);
    num_heads_kv = json_data.value("num_key_value_heads", num_heads_kv);
    head_dim = json_data.value("head_dim", head_dim);
    hidden_size = json_data.value("hidden_size", hidden_size);
    vocab_size = json_data.value("vocab_size", vocab_size);
    max_len = json_data.value("max_position_embeddings", max_len);
    bos_token_id = json_data.value("bos_token_id", bos_token_id);
    eos_token_id = json_data.value("eos_token_id", eos_token_id);
    rope_theta = json_data.value("rope_theta", rope_theta);
    attention_bias = json_data.value("attention_bias", attention_bias);
    model_type = json_data.value("model_type", model_type);
    if (json_data.contains("architectures") &&
        json_data["architectures"].is_array() &&
        !json_data["architectures"].empty() &&
        json_data["architectures"][0].is_string()) {
        architecture = json_data["architectures"][0].get<std::string>();
    }

    if (head_dim <= 0 && num_heads > 0 && hidden_size > 0 && hidden_size % num_heads == 0) {
        head_dim = hidden_size / num_heads;
    }

    spdlog::info(
        "Loaded model config: layers={}, heads={}, kv_heads={}, head_dim={}, hidden_size={}, vocab_size={}, max_len={}, bos={}, eos={}, rope_theta={}, attention_bias={}, architecture={}, model_type={}",
        num_layers,
        num_heads,
        num_heads_kv,
        head_dim,
        hidden_size,
        vocab_size,
        max_len,
        bos_token_id,
        eos_token_id,
        rope_theta,
        attention_bias,
        architecture,
        model_type
    );
}

} // namespace easy_llm
