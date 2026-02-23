#include "spdlog/spdlog.h"

#include <iostream>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cli_options.hpp"
#include "continuous_batch_server.hpp"
#include "gpt_engine.hpp"
#include "config.hpp"
#include "data_manager.hpp"
#include "tokenizer.hpp"
#include "models/gpt_model.hpp"
#include "models/loader.hpp"

using std::make_unique;
using std::move;
using std::string;
using std::vector;

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("start gpt_engine running");

    easy_llm::CliOptions options;
    string parse_error;
    if (!easy_llm::parse_args(argc, argv, &options, &parse_error)) {
        spdlog::error(parse_error);
        easy_llm::print_usage(std::cerr);
        return 1;
    }
    if (options.show_help) {
        easy_llm::print_usage(std::cout);
        return 0;
    }

    vector<string> prompts;
    if (!options.prompt_file.empty()) {
        string file_error;
        if (!easy_llm::read_prompts_file(options.prompt_file, &prompts, &file_error)) {
            spdlog::error(file_error);
            return 1;
        }
    }
    if (!options.prompt.empty()) {
        prompts.push_back(move(options.prompt));
    }
    if (prompts.empty() && !options.serve) {
        spdlog::error("No prompts provided");
        easy_llm::print_usage(std::cerr);
        return 1;
    }

    auto config = make_unique<easy_llm::Config>();
    if (!options.model_dir.empty()) {
        const std::filesystem::path model_dir = options.model_dir;
        config->model_config_path = (model_dir / "config.json").string();
        config->model_path = (model_dir / "model.safetensors").string();
        config->tokenizer_path = (model_dir / "tokenizer.json").string();
        config->tokenizer_config_path = (model_dir / "tokenizer_config.json").string();
    }
    config->load_config();
    config->max_steps = options.max_steps;
    config->temperature = options.temperature;
    config->top_p = options.top_p;
    config->top_k = options.top_k;
    config->seed = options.seed;
    config->use_greedy = options.use_greedy;
    auto model_param = easy_llm::ModelParam::load(config->model_path);
    auto tokenizer_for_batch = easy_llm::Tokenizer::create(*config);
    auto data_manager = make_unique<easy_llm::DataManager>(move(tokenizer_for_batch));
    auto gpt_model = easy_llm::GptModel::create(*config, *data_manager, *model_param);

    for (string& prompt : prompts) {
        prompt = easy_llm::apply_chat_template(prompt);
    }

    if (options.serve) {
        auto service_tokenizer = easy_llm::Tokenizer::create(*config);
        easy_llm::ContinuousBatchOptions server_options;
        server_options.max_active_requests = options.serve_max_active;
        server_options.prefill_batch_size = options.serve_prefill_batch;
        server_options.idle_sleep_ms = options.serve_idle_ms;
        server_options.stats_log_interval_ms = options.serve_stats_ms;
        easy_llm::ContinuousBatchServer server(*config, *gpt_model, *service_tokenizer, server_options);
        for (const auto& prompt : prompts) {
            server.submit_prompt(prompt);
        }

        std::thread input_thread([&server]() {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "/quit" || line == ":quit") {
                    break;
                }
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                    continue;
                }
                std::string templated = easy_llm::apply_chat_template(line);
                int request_id = server.submit_prompt(std::move(templated));
                if (request_id >= 0) {
                    std::cout << "[accepted " << request_id << "]\n";
                    std::cout.flush();
                }
            }
            server.mark_input_closed();
        });

        std::cout << "continuous batching service started; submit one prompt per line, /quit to stop\n";
        std::cout.flush();
        server.run();
        input_thread.join();
        spdlog::info("end gpt_engine running");
        return 0;
    }

    std::string output_path;
    if (!options.prompt_file.empty()) {
        std::filesystem::path prompt_path(options.prompt_file);
        const std::string stem = prompt_path.stem().string();
        const std::string ext = prompt_path.extension().string();
        const std::string output_name = stem + "_output" + ext;
        output_path = (prompt_path.parent_path() / output_name).string();
    }
    easy_llm::GptEngine gpt_engine{move(gpt_model), move(config), move(data_manager)};
    gpt_engine.run(prompts, output_path);
    spdlog::info("end gpt_engine running");
    return 0;
}
