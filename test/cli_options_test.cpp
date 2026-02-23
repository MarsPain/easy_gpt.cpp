#include <iostream>
#include <string>

#include "cli_options.hpp"

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int count_substring(const std::string& text, const std::string& needle) {
    if (needle.empty()) {
        return 0;
    }
    int count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int run_valid_case() {
    easy_llm::CliOptions options;
    std::string error;
    const char* argv[] = {
        "easy_llm",
        "--serve",
        "--serve-stats-ms",
        "2500",
        "--serve-idle-ms",
        "3",
        "--serve-max-active",
        "2"
    };
    const int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));
    if (!easy_llm::parse_args(argc, const_cast<char**>(argv), &options, &error)) {
        std::cerr << "FAIL: valid args should parse, error=" << error << "\n";
        return 1;
    }
    if (!options.serve) {
        std::cerr << "FAIL: --serve not parsed\n";
        return 1;
    }
    if (options.serve_stats_ms != 2500) {
        std::cerr << "FAIL: --serve-stats-ms parsed value mismatch\n";
        return 1;
    }
    if (options.serve_idle_ms != 3 || options.serve_max_active != 2) {
        std::cerr << "FAIL: unrelated serve args parsed incorrectly\n";
        return 1;
    }
    return 0;
}

int run_model_dir_case() {
    easy_llm::CliOptions options;
    std::string error;
    const char* argv[] = {
        "easy_llm",
        "--model-dir",
        "/tmp/qwen3"
    };
    const int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));
    if (!easy_llm::parse_args(argc, const_cast<char**>(argv), &options, &error)) {
        std::cerr << "FAIL: --model-dir should parse, error=" << error << "\n";
        return 1;
    }
    if (options.model_dir != "/tmp/qwen3") {
        std::cerr << "FAIL: --model-dir parsed value mismatch\n";
        return 1;
    }
    return 0;
}

int run_invalid_case() {
    easy_llm::CliOptions options;
    std::string error;
    const char* argv[] = {
        "easy_llm",
        "--serve-stats-ms",
        "-1"
    };
    const int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));
    if (easy_llm::parse_args(argc, const_cast<char**>(argv), &options, &error)) {
        std::cerr << "FAIL: negative --serve-stats-ms should fail\n";
        return 1;
    }
    if (!contains(error, "serve-stats-ms")) {
        std::cerr << "FAIL: error message should mention serve-stats-ms, got=" << error << "\n";
        return 1;
    }
    return 0;
}

int run_chat_template_qwen3_no_think_case() {
    const std::string templated = easy_llm::apply_chat_template("Hello", "qwen3");
    if (contains(templated, "<|im_start|>system\n")) {
        std::cerr << "FAIL: qwen3 template should not inject default system message\n";
        return 1;
    }
    if (!contains(templated, "<|im_start|>user\nHello\n/no_think<|im_end|>\n<|im_start|>assistant\n")) {
        std::cerr << "FAIL: qwen3 template should follow user->assistant format\n";
        return 1;
    }
    if (!contains(templated, "Hello\n/no_think<|im_end|>")) {
        std::cerr << "FAIL: qwen3 template should append /no_think\n";
        return 1;
    }

    const std::string templated_existing =
        easy_llm::apply_chat_template("Hello\n/no_think", "Qwen3ForCausalLM");
    if (count_substring(templated_existing, "/no_think") != 1) {
        std::cerr << "FAIL: qwen3 template should not duplicate /no_think\n";
        return 1;
    }
    return 0;
}

int run_chat_template_non_qwen3_case() {
    const std::string templated = easy_llm::apply_chat_template("Hello", "qwen2");
    if (!contains(templated, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n")) {
        std::cerr << "FAIL: qwen2.5 template should inject default system message\n";
        return 1;
    }
    if (!contains(templated, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n")) {
        std::cerr << "FAIL: qwen2.5 template should follow system->user->assistant format\n";
        return 1;
    }
    if (contains(templated, "/no_think")) {
        std::cerr << "FAIL: non-qwen3 template should not append /no_think\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (run_valid_case() != 0) {
        return 1;
    }
    if (run_model_dir_case() != 0) {
        return 1;
    }
    if (run_invalid_case() != 0) {
        return 1;
    }
    if (run_chat_template_qwen3_no_think_case() != 0) {
        return 1;
    }
    if (run_chat_template_non_qwen3_case() != 0) {
        return 1;
    }
    std::cout << "PASS: cli_options test\n";
    return 0;
}
