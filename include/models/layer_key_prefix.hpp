#ifndef EASY_LLM_MODELS_LAYER_KEY_PREFIX_HPP
#define EASY_LLM_MODELS_LAYER_KEY_PREFIX_HPP

#include <memory>
#include <string>

namespace easy_llm {

struct Config;

class LayerKeyPrefix {
public:
    virtual ~LayerKeyPrefix() = default;

    virtual std::string layer(int layer_idx) const = 0;
    virtual std::string model_norm() const = 0;

    virtual std::string self_attn_q_proj(const std::string& layer_key) const = 0;
    virtual std::string self_attn_k_proj(const std::string& layer_key) const = 0;
    virtual std::string self_attn_v_proj(const std::string& layer_key) const = 0;
    virtual std::string self_attn_o_proj(const std::string& layer_key) const = 0;
    virtual std::string self_attn_q_norm(const std::string& layer_key) const = 0;
    virtual std::string self_attn_k_norm(const std::string& layer_key) const = 0;
    virtual bool uses_qk_norm() const = 0;
    virtual std::string input_layer_norm(const std::string& layer_key) const = 0;

    virtual std::string mlp_down_proj(const std::string& layer_key) const = 0;
    virtual std::string mlp_gate_proj(const std::string& layer_key) const = 0;
    virtual std::string mlp_up_proj(const std::string& layer_key) const = 0;
    virtual std::string post_attention_layer_norm(const std::string& layer_key) const = 0;
};

std::unique_ptr<LayerKeyPrefix> create_layer_key_prefix(const Config& config);

} // namespace easy_llm

#endif // EASY_LLM_MODELS_LAYER_KEY_PREFIX_HPP
