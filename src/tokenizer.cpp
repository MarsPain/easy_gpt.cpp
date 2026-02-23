#include "tokenizer.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <algorithm>
#include <stdexcept>

#include "spdlog/spdlog.h"
#include "json.hpp"
#include "bpe.hpp"

#include "config.hpp"
#include "utils.hpp"

namespace easy_llm {

using std::string;
using std::vector;

namespace {

struct SpecialMatch {
    size_t pos{string::npos};
    size_t len{0};
    string token;
};

class EncodingSession {
public:
    EncodingSession(const Bpe& bpe, const vector<string>& special_tokens, const string& text)
        : bpe_(bpe), special_tokens_(special_tokens), text_(text), tokens_() {}

    vector<string> run() {
        if (special_tokens_.empty()) {
            return bpe_.encode(text_);
        }
        size_t pos = 0;
        while (pos < text_.size()) {
            SpecialMatch match;
            if (!find_next_special_token(pos, match)) {
                encode_plain_segment(pos, text_.size() - pos);
                break;
            }
            if (match.pos > pos) {
                encode_plain_segment(pos, match.pos - pos);
            }
            tokens_.push_back(match.token);
            suppress_prefix_next_ = compute_suppress_prefix_next(match.pos, match.len);
            pos = match.pos + match.len;
        }
        return tokens_;
    }

private:
    bool find_next_special_token(size_t start_pos, SpecialMatch& match) const {
        match.pos = string::npos;
        match.len = 0;
        match.token.clear();
        for (const auto& special : special_tokens_) {
            size_t found = text_.find(special, start_pos);
            if (found == string::npos) {
                continue;
            }
            if (match.pos == string::npos || found < match.pos ||
                (found == match.pos && special.size() > match.len)) {
                match.pos = found;
                match.len = special.size();
                match.token = special;
            }
        }
        return match.pos != string::npos;
    }

    void encode_plain_segment(size_t start_pos, size_t len) {
        if (len == 0) {
            return;
        }
        bpe_.encode_into(text_.substr(start_pos, len), first_word_, suppress_prefix_next_, tokens_);
    }

    bool compute_suppress_prefix_next(size_t match_pos, size_t match_len) const {
        size_t after = match_pos + match_len;
        bool left_non_ws = match_pos > 0 &&
            !std::isspace(static_cast<unsigned char>(text_[match_pos - 1]));
        bool right_non_ws = after < text_.size() &&
            !std::isspace(static_cast<unsigned char>(text_[after]));
        return left_non_ws && right_non_ws;
    }

    const Bpe& bpe_;
    const vector<string>& special_tokens_;
    const string& text_;
    vector<string> tokens_;
    bool first_word_{true};
    bool suppress_prefix_next_{false};
};

std::vector<std::string> parse_token_merges(const nlohmann::json& tokenizer_config) {
    if (!tokenizer_config.contains("model") || !tokenizer_config["model"].is_object() ||
        !tokenizer_config["model"].contains("merges") || !tokenizer_config["model"]["merges"].is_array()) {
        throw std::runtime_error("Tokenizer JSON missing model.merges array.");
    }
    std::vector<std::string> token_merges;
    const auto& merges = tokenizer_config["model"]["merges"];
    token_merges.reserve(merges.size());
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const auto& merge = merges[i];
        if (merge.is_string()) {
            token_merges.push_back(merge.get<std::string>());
            continue;
        }
        if (merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string()) {
            token_merges.push_back(merge[0].get<std::string>() + " " + merge[1].get<std::string>());
            continue;
        }
        throw std::runtime_error("Unsupported merge format at tokenizer model.merges[" + std::to_string(i) + "].");
    }
    return token_merges;
}

} // namespace

std::unique_ptr<Tokenizer> Tokenizer::create(const Config& config) {
    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->init_from_config(config);
    spdlog::info("Tokenizer created");
    return tokenizer;
}

void Tokenizer::init_from_config(const Config& config) {
    bos_token_id_ = config.bos_token_id;
    init_tokenizer_config(config.tokenizer_path);
    init_special_tokens(config.tokenizer_config_path);
}

vector<string> Tokenizer::tokenize(const string& text) const {
    EncodingSession session(*bpe_, special_tokens_, text);
    auto tokens = session.run();
    spdlog::info("tokens_joined == {} | decoded == {}", fmt::join(tokens, " "),
                 bpe_->decode(tokens));
    return tokens;
}

string Tokenizer::decode(const vector<int>& ids) const {
    auto tokens = ids_to_tokens(ids);
    return bpe_->decode(tokens);
}

vector<int> Tokenizer::tokens_to_ids(const vector<string>& tokens) const {
    vector<int> token_ids{};
    token_ids.reserve(tokens.size() + (bos_token_id_ >= 0 ? 1U : 0U));
    if (bos_token_id_ >= 0) {
        token_ids.push_back(bos_token_id_);
    }
    for (const string& token : tokens) {
        auto it = token2id_.find(token);
        if (it != token2id_.end()) {
            token_ids.push_back(it->second);
        } else {
            spdlog::warn("token {} not in vocab", token);
        }
    }
    return token_ids;
}

vector<string> Tokenizer::ids_to_tokens(const vector<int>& ids) const {
    vector<string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        auto it = id2token_.find(id);
        if (it != id2token_.end()) {
            tokens.push_back(it->second);
        } else {
            spdlog::warn("id {} not in vocab", id);
        }
    }
    return tokens;
}

void Tokenizer::init_tokenizer_config(const string& tokenizer_path) {
    tokenizer_config_ = utils::load_json(tokenizer_path);
    token_merges_ = parse_token_merges(tokenizer_config_);
    spdlog::debug("size of token_merges: {}", token_merges_.size());
    bpe_ = std::make_unique<Bpe>(token_merges_);
    token2id_ = tokenizer_config_["model"]["vocab"];
    spdlog::debug("size of token2id_ : {}", token2id_.size());
    for (auto& kv : token2id_) {
        id2token_[kv.second] = kv.first;
    }
}

void Tokenizer::init_special_tokens(const string& tokenizer_config_path) {
    auto tokenizer_config = utils::load_json(tokenizer_config_path);
    if (tokenizer_config.empty()) {
        spdlog::error("Failed to load tokenizer_config.json: {}", tokenizer_config_path);
        return;
    }

    special_tokens_.clear();
    auto added_tokens = parse_added_tokens(tokenizer_config);
    register_added_tokens(added_tokens);
    if (!parse_pad_id(tokenizer_config)) {
        return;
    }
    sort_special_tokens();
}

std::vector<Tokenizer::AddedTokenInfo> Tokenizer::parse_added_tokens(
    const nlohmann::json& tokenizer_config) const {
    std::vector<AddedTokenInfo> added_tokens;
    if (!tokenizer_config.contains("added_tokens_decoder")) {
        return added_tokens;
    }

    for (auto& [id_str, token_info] : tokenizer_config["added_tokens_decoder"].items()) {
        if (!token_info.contains("content")) {
            spdlog::warn("added_tokens_decoder entry missing content for id {}", id_str);
            continue;
        }
        const string token = token_info["content"].get<string>();
        if (token.empty()) {
            spdlog::warn("added_tokens_decoder entry has empty content for id {}", id_str);
            continue;
        }

        int id = -1;
        try {
            id = std::stoi(id_str);
        } catch (const std::exception& e) {
            spdlog::warn("invalid added_tokens_decoder id '{}': {}", id_str, e.what());
            continue;
        }
        bool is_special = token_info.value("special", false);
        added_tokens.push_back({id, token, is_special});
    }
    return added_tokens;
}

void Tokenizer::register_added_tokens(const std::vector<AddedTokenInfo>& added_tokens) {
    if (added_tokens.empty()) {
        return;
    }

    std::unordered_set<string> seen_special_tokens;
    for (const auto& info : added_tokens) {
        if (info.is_special && seen_special_tokens.insert(info.token).second) {
            special_tokens_.push_back(info.token);
        }

        bool token_exists = token2id_.find(info.token) != token2id_.end();
        bool id_exists = id2token_.find(info.id) != id2token_.end();
        if (token_exists || id_exists) {
            spdlog::warn(
                "added_tokens_decoder token/id already in vocab: token='{}' id={} token_exists={} id_exists={}",
                info.token, info.id, token_exists, id_exists);
            continue;
        }
        token2id_[info.token] = info.id;
        id2token_[info.id] = info.token;
    }
}

bool Tokenizer::parse_pad_id(const nlohmann::json& tokenizer_config) {
    std::string pad_token;
    if (tokenizer_config.contains("pad_token")) {
        const auto& pad_token_json = tokenizer_config["pad_token"];
        if (pad_token_json.is_string()) {
            pad_token = pad_token_json.get<std::string>();
        } else if (pad_token_json.is_object() && pad_token_json.contains("content") &&
                   pad_token_json["content"].is_string()) {
            pad_token = pad_token_json["content"].get<std::string>();
        } else {
            spdlog::warn("pad_token has unsupported json type in tokenizer_config.json");
        }
    }
    if (pad_token.empty()) {
        spdlog::warn("pad_token not found in tokenizer_config.json");
        return false;
    }
    if (tokenizer_config.contains("added_tokens_decoder")) {
        for (auto& [id_str, token_info] : tokenizer_config["added_tokens_decoder"].items()) {
            if (token_info.contains("content") && token_info["content"] == pad_token) {
                pad_id_ = std::stoi(id_str);
                break;
            }
        }
    }

    if (pad_id_ < 0) {
        auto it = token2id_.find(pad_token);
        if (it != token2id_.end()) {
            pad_id_ = it->second;
        }
    }
    if (pad_id_ < 0) {
        spdlog::warn("pad_token '{}' not found in tokenizer data", pad_token);
    }
    return true;
}

void Tokenizer::sort_special_tokens() {
    if (special_tokens_.empty()) {
        return;
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const string& a, const string& b) {
                  if (a.size() != b.size()) {
                      return a.size() > b.size();
                  }
                  return a < b;
              });
}

} // namespace easy_llm
