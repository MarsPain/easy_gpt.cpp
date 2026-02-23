#include "cli_options.hpp"

#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace easy_llm {
namespace {

std::string trim(const std::string& text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(start, end - start + 1);
}

}  // namespace

void print_usage(std::ostream& os) {
    os << "Usage: easy_llm [--model-dir <path>] [--prompt-file <path>] [--max-steps <n>]\n"
          "                [--temperature <float>] [--top-p <float>] [--top-k <int>] [--seed <int>] [--greedy]\n"
          "                [--serve] [--serve-max-active <n>] [--serve-prefill-batch <n>] [--serve-idle-ms <n>] [--serve-stats-ms <n>]\n"
          "                [\"prompt\"]\n"
       << "      --model-dir <path>    Override model directory containing config/model/tokenizer files\n"
       << "  -f, --prompt-file <path>  Read prompts from file (one per line, ignore empty lines)\n"
       << "  -m, --max-steps <n>        Maximum generation steps per request (default: 100)\n"
       << "      --temperature <float> Sampling temperature (default: 0.8)\n"
       << "      --top-p <float>        Nucleus sampling cutoff in (0,1] (default: 0.95)\n"
       << "      --top-k <int>          Top-K sampling cutoff, 0 disables (default: 20)\n"
       << "      --seed <int>           RNG seed for sampling (default: 42)\n"
       << "      --greedy               Use greedy decoding (override sampling)\n"
       << "      --serve                Run as long-lived continuous batching service\n"
       << "      --serve-max-active <n> Max concurrent active requests in service mode (default: 16)\n"
       << "      --serve-prefill-batch <n> Max requests admitted per prefill round (default: 4)\n"
       << "      --serve-idle-ms <n>    Service idle sleep in milliseconds (default: 2)\n"
       << "      --serve-stats-ms <n>   Service stats log interval in milliseconds, 0 disables periodic logs (default: 1000)\n"
       << "  -h, --help                Show this help message\n"
       << "Examples:\n"
       << "  ./build/easy_llm --model-dir model/Qwen3-0.6B --max-steps 128 \"Hello\"\n"
       << "  ./build/easy_llm --max-steps 128 \"Hello\"\n"
       << "  ./build/easy_llm --temperature 0.7 --top-p 0.9 --top-k 40 \"Hello\"\n"
       << "  ./build/easy_llm --greedy \"Hello\"\n"
       << "  ./build/easy_llm -f test/data/prompts.txt\n"
       << "  ./build/easy_llm --serve\n";
}

std::string apply_chat_template(const std::string& user_query) {
    static const std::string kChatTemplate =
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "{user_query}<|im_end|>\n"
        "<|im_start|>assistant\n";
    std::string result = kChatTemplate;
    const std::string token = "{user_query}";
    size_t pos = 0;
    while ((pos = result.find(token, pos)) != std::string::npos) {
        result.replace(pos, token.size(), user_query);
        pos += user_query.size();
    }
    return result;
}

bool read_prompts_file(const std::string& path, std::vector<std::string>* prompts, std::string* error) {
    std::ifstream in(path);
    if (!in.is_open()) {
        *error = "Failed to open prompt file: " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::string trimmed = trim(line);
        if (!trimmed.empty()) {
            prompts->push_back(std::move(trimmed));
        }
    }
    if (in.bad()) {
        *error = "Failed while reading prompt file: " + path;
        return false;
    }
    return true;
}

bool parse_args(int argc, char** argv, CliOptions* options, std::string* error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            options->show_help = true;
            return true;
        }
        if (arg == "-f" || arg == "--prompt-file") {
            if (i + 1 >= argc) {
                *error = "Missing path after " + arg;
                return false;
            }
            if (!options->prompt_file.empty()) {
                *error = "Prompt file option specified multiple times";
                return false;
            }
            options->prompt_file = argv[++i];
            continue;
        }
        if (arg == "--model-dir") {
            if (i + 1 >= argc) {
                *error = "Missing path after " + arg;
                return false;
            }
            if (!options->model_dir.empty()) {
                *error = "Model dir option specified multiple times";
                return false;
            }
            options->model_dir = argv[++i];
            continue;
        }
        if (arg == "-m" || arg == "--max-steps") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->max_steps = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for max steps: " + value;
                return false;
            }
            if (options->max_steps <= 0) {
                *error = "Max steps must be greater than 0";
                return false;
            }
            continue;
        }
        if (arg == "--temperature") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->temperature = std::stof(value);
            } catch (const std::exception&) {
                *error = "Invalid number for temperature: " + value;
                return false;
            }
            if (options->temperature <= 0.0f) {
                *error = "Temperature must be greater than 0";
                return false;
            }
            continue;
        }
        if (arg == "--top-p") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->top_p = std::stof(value);
            } catch (const std::exception&) {
                *error = "Invalid number for top-p: " + value;
                return false;
            }
            if (options->top_p <= 0.0f || options->top_p > 1.0f) {
                *error = "Top-p must be in (0, 1]";
                return false;
            }
            continue;
        }
        if (arg == "--top-k") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->top_k = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for top-k: " + value;
                return false;
            }
            if (options->top_k < 0) {
                *error = "Top-k must be >= 0";
                return false;
            }
            continue;
        }
        if (arg == "--seed") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->seed = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for seed: " + value;
                return false;
            }
            if (options->seed < 0) {
                *error = "Seed must be >= 0";
                return false;
            }
            continue;
        }
        if (arg == "--greedy") {
            options->use_greedy = true;
            continue;
        }
        if (arg == "--serve") {
            options->serve = true;
            continue;
        }
        if (arg == "--serve-max-active") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->serve_max_active = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for serve max active: " + value;
                return false;
            }
            if (options->serve_max_active <= 0) {
                *error = "serve-max-active must be greater than 0";
                return false;
            }
            continue;
        }
        if (arg == "--serve-prefill-batch") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->serve_prefill_batch = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for serve prefill batch: " + value;
                return false;
            }
            if (options->serve_prefill_batch <= 0) {
                *error = "serve-prefill-batch must be greater than 0";
                return false;
            }
            continue;
        }
        if (arg == "--serve-idle-ms") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->serve_idle_ms = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for serve idle ms: " + value;
                return false;
            }
            if (options->serve_idle_ms < 0) {
                *error = "serve-idle-ms must be >= 0";
                return false;
            }
            continue;
        }
        if (arg == "--serve-stats-ms") {
            if (i + 1 >= argc) {
                *error = "Missing number after " + arg;
                return false;
            }
            std::string value = argv[++i];
            try {
                options->serve_stats_ms = std::stoi(value);
            } catch (const std::exception&) {
                *error = "Invalid number for serve stats ms: " + value;
                return false;
            }
            if (options->serve_stats_ms < 0) {
                *error = "serve-stats-ms must be >= 0";
                return false;
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            *error = "Unknown option: " + arg;
            return false;
        }
        if (!options->prompt.empty()) {
            *error = "Multiple prompts provided; use a single prompt or --prompt-file";
            return false;
        }
        options->prompt = std::move(arg);
    }
    return true;
}

}  // namespace easy_llm
