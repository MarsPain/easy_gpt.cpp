#ifndef EASY_LLM_CONFIG_HPP
#define EASY_LLM_CONFIG_HPP

#include <string>

namespace easy_llm {

struct Config {
    void load_config();

    int num_layers{0};
    int num_heads{0};
    int num_heads_kv{0};
    int head_dim{0};
    int hidden_size{0};
    int vocab_size{0};
    int max_len{0};
    int max_steps{100};
    int batch_size{0};
    int num_samples{0};
    int bos_token_id{-1};
    int eos_token_id{-1};
    float rope_theta{0.0f};
    bool attention_bias{true};
    float temperature{1.0f};
    float top_p{1.0f};
    int top_k{0};
    int seed{42};
    bool use_greedy{false};
    std::string architecture;
    std::string model_type;

    std::string data_path;
    std::string model_config_path{"data/model/config.json"};
    std::string model_path{"data/model/model.safetensors"};
    std::string tokenizer_path{"data/model/tokenizer.json"};
    std::string tokenizer_config_path{"data/model/tokenizer_config.json"};
    std::string vocab_path;
    std::string log_path;
    std::string save_path;
    std::string load_path;

};

} // namespace easy_llm

#endif // EASY_LLM_CONFIG_HPP
