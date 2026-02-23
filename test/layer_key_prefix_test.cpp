#include <iostream>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "models/layer_key_prefix.hpp"

namespace {

template <typename ExceptionType, typename Fn>
bool expect_throws(Fn&& fn, const char* case_name) {
    try {
        fn();
    } catch (const ExceptionType&) {
        return true;
    }
    std::cerr << "FAIL: " << case_name << " should throw\n";
    return false;
}

} // namespace

int main() {
    {
        easy_llm::Config config;
        config.architecture = "Qwen2ForCausalLM";
        config.model_type = "qwen2";
        auto prefix = easy_llm::create_layer_key_prefix(config);
        if (!prefix) {
            std::cerr << "FAIL: qwen2 prefix should be created\n";
            return 1;
        }
        if (prefix->layer(3) != "model.layers.3") {
            std::cerr << "FAIL: qwen2 layer key mismatch\n";
            return 1;
        }
        if (prefix->uses_qk_norm()) {
            std::cerr << "FAIL: qwen2 should not enable qk norm\n";
            return 1;
        }
    }

    {
        easy_llm::Config config;
        config.architecture = "Qwen3ForCausalLM";
        config.model_type = "qwen3";
        auto prefix = easy_llm::create_layer_key_prefix(config);
        if (!prefix) {
            std::cerr << "FAIL: qwen3 prefix should be created\n";
            return 1;
        }
        if (!prefix->uses_qk_norm()) {
            std::cerr << "FAIL: qwen3 should enable qk norm\n";
            return 1;
        }
        const std::string layer_key = prefix->layer(7);
        if (prefix->self_attn_q_norm(layer_key) != "model.layers.7.self_attn.q_norm") {
            std::cerr << "FAIL: qwen3 q_norm key mismatch\n";
            return 1;
        }
        if (prefix->self_attn_k_norm(layer_key) != "model.layers.7.self_attn.k_norm") {
            std::cerr << "FAIL: qwen3 k_norm key mismatch\n";
            return 1;
        }
    }

    {
        easy_llm::Config config;
        config.architecture = "UnknownArchForCausalLM";
        config.model_type = "unknown";
        if (!expect_throws<std::invalid_argument>([&]() {
                (void)easy_llm::create_layer_key_prefix(config);
            }, "unknown model dispatch")) {
            return 1;
        }
    }

    std::cout << "PASS: layer key prefix test\n";
    return 0;
}
