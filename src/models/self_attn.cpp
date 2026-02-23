#include "models/self_attn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include "ops.hpp"

#ifdef USE_CUDA
    #include "cuda/runtime.hpp"
#endif

#include "config.hpp"
#include "models/layer_key_prefix.hpp"

namespace easy_llm {

namespace {

int shape_product(const std::vector<int>& shape) {
    int total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    return total;
}

void validate_tensor_size(const Tensor& tensor, const char* name) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument(std::string("Tensor has empty shape: ") + name);
    }
    int expected = shape_product(tensor.shape());
    if (tensor.size() != expected) {
        throw std::invalid_argument(std::string("Tensor size mismatch: ") + name);
    }
}

Tensor slice_batch_4d(const Tensor& input, int batch_index) {
    const auto& shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("slice_batch_4d expects a 4D tensor.");
    }
    int batch = shape[0];
    if (batch_index < 0 || batch_index >= batch) {
        throw std::out_of_range("slice_batch_4d batch index out of range.");
    }
    int num_heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    int block = num_heads * seq_len * head_dim;
    if (input.size() != batch * block) {
        throw std::invalid_argument("slice_batch_4d: input size does not match shape.");
    }
    int start = batch_index * block;
    std::vector<data_type> data(static_cast<size_t>(block));
    const auto& src = input.data();
    std::copy(src.begin() + start, src.begin() + start + block, data.begin());
    return Tensor{data, {1, num_heads, seq_len, head_dim}};
}

bool is_uniform_offset(const std::vector<int>& offsets) {
    if (offsets.empty()) {
        return true;
    }
    int offset0 = offsets[0];
    for (int offset : offsets) {
        if (offset != offset0) {
            return false;
        }
    }
    return true;
}

void apply_causal_mask(Tensor& scores, int seq_len, int total_len) {
    int batch = scores.shape()[0];
    int heads = scores.shape()[1];
    float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();
    int start_pos = total_len - seq_len;

#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int mask_start = start_pos + s + 1;
                if (mask_start < total_len) {
                    int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                    for (int k = mask_start; k < total_len; ++k) {
                        data[row_offset + k] = data_type(neg_inf);
                    }
                }
            }
        }
    }
}

void apply_padding_mask(Tensor& scores, const std::vector<int>& sample_ids, const std::vector<int>& pad_lens) {
    int batch = scores.shape()[0];
    int heads = scores.shape()[1];
    int seq_len = scores.shape()[2];
    int total_len = scores.shape()[3];
    float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();

    for (int b = 0; b < batch; ++b) {
        int sample_id = sample_ids[b];
        if (sample_id < 0 || sample_id >= static_cast<int>(pad_lens.size())) {
            continue;
        }
        int pad_len = pad_lens[sample_id];
        if (pad_len <= 0) {
            continue;
        }
        if (pad_len > total_len) {
            pad_len = total_len;
        }
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                for (int k = 0; k < pad_len; ++k) {
                    data[row_offset + k] = data_type(neg_inf);
                }
            }
        }
    }
}

} // namespace

struct SelfAttn::ForwardContext {
    const Tensor& input;
    const std::vector<int>& sample_ids;
    const std::vector<int>* pos_offsets;
    std::vector<int> offsets;
    int seq_len{0};
    int max_sample_id{-1};

    ForwardContext(const Tensor& input,
                   const std::vector<int>& sample_ids,
                   const std::vector<int>* pos_offsets)
        : input(input), sample_ids(sample_ids), pos_offsets(pos_offsets) {}
};

SelfAttn::SelfAttn() {}

SelfAttn::SelfAttn(const Config& config) {
    num_heads_ = config.num_heads;
    num_heads_kv_ = config.num_heads_kv;
    rope_theta_ = config.rope_theta;
}

SelfAttn::SelfAttn(int hidden_dim, int num_heads)
    : hidden_dim_(hidden_dim), num_heads_(num_heads),
      head_dim_(hidden_dim / num_heads),
      q_proj_(hidden_dim, hidden_dim),
      k_proj_(hidden_dim, hidden_dim),
      v_proj_(hidden_dim, hidden_dim),
      o_proj_(hidden_dim, hidden_dim) {
}

void SelfAttn::load_param(const LayerKeyPrefix& key_prefix, const std::string& key, ModelParam& model_param) {
    use_qk_norm_ = false;
#ifdef USE_CUDA
    cuda_supported_ = true;
#endif
    q_proj_.load_param(key_prefix.self_attn_q_proj(key), model_param);
    k_proj_.load_param(key_prefix.self_attn_k_proj(key), model_param);
    v_proj_.load_param(key_prefix.self_attn_v_proj(key), model_param);
    o_proj_.load_param(key_prefix.self_attn_o_proj(key), model_param);
    hidden_dim_ = q_proj_.get_out_dim();
    if (num_heads_ <= 0) {
        spdlog::error("num_heads_ is invalid: {}", num_heads_);
        return;
    }
    head_dim_ = hidden_dim_ / num_heads_;
    norm_.load_param(key_prefix.input_layer_norm(key), model_param);

    const std::string q_norm_key = key_prefix.self_attn_q_norm(key);
    const std::string k_norm_key = key_prefix.self_attn_k_norm(key);
    const bool has_q_norm = model_param.contains(q_norm_key + ".weight");
    const bool has_k_norm = model_param.contains(k_norm_key + ".weight");

    if (key_prefix.uses_qk_norm()) {
        if (!has_q_norm || !has_k_norm) {
            throw std::runtime_error("QK-Norm is required but missing weights under key: " + key);
        }
        q_norm_.load_param(q_norm_key, model_param);
        k_norm_.load_param(k_norm_key, model_param);
        use_qk_norm_ = true;
    } else if (has_q_norm && has_k_norm) {
        q_norm_.load_param(q_norm_key, model_param);
        k_norm_.load_param(k_norm_key, model_param);
        use_qk_norm_ = true;
    } else if (has_q_norm != has_k_norm) {
        throw std::runtime_error("QK-Norm weights are partially present under key: " + key);
    }

#ifdef USE_CUDA
    if (use_qk_norm_) {
        cuda_supported_ = false;
        spdlog::warn("SelfAttn CUDA disabled for layer '{}' because QK-Norm is enabled.", key);
    }
#endif
}

Tensor SelfAttn::forward(const Tensor& input, const std::vector<int>& sample_ids, const std::vector<int>* pos_offsets) {
    if (num_heads_ <= 0 || num_heads_kv_ <= 0 || rope_theta_ <= 0.0f) {
        spdlog::error("SelfAttn not configured: num_heads={}, num_heads_kv={}, rope_theta={}", num_heads_, num_heads_kv_, rope_theta_);
        return Tensor();
    }
    ForwardContext ctx(input, sample_ids, pos_offsets);
    validate_forward_inputs(ctx);
    ctx.max_sample_id = *std::max_element(ctx.sample_ids.begin(), ctx.sample_ids.end());
    ensure_cache_capacity(ctx.max_sample_id + 1);

    compute_offsets(ctx);

#ifdef USE_CUDA
    if (cuda_enabled_ && cuda_supported_ && ::easy_llm::cuda::available()) {
        try {
            auto output = forward_cuda(input, sample_ids, ctx.offsets);
            return output;
        } catch (const std::exception& e) {
            bool has_cuda_cache = false;
            if (cuda_state_) {
                for (int i = 0; i <= ctx.max_sample_id; ++i) {
                    if (cuda_state_->cache_len(i) > 0) {
                        has_cuda_cache = true;
                        break;
                    }
                }
            }
            if (has_cuda_cache) {
                spdlog::error("SelfAttn CUDA forward failed with active CUDA KV cache. Fallback is unsafe: {}", e.what());
                throw;
            }
            cuda_enabled_ = false;
            spdlog::error("SelfAttn CUDA forward failed. Falling back to CPU path: {}", e.what());
        }
    }
#endif
    return forward_cpu(input, sample_ids, pos_offsets);
}

Tensor SelfAttn::forward_cpu(const Tensor& input, const std::vector<int>& sample_ids, const std::vector<int>* pos_offsets) {
    ForwardContext ctx(input, sample_ids, pos_offsets);
    validate_forward_inputs(ctx);
    ctx.max_sample_id = *std::max_element(ctx.sample_ids.begin(), ctx.sample_ids.end());
    ensure_cache_capacity(ctx.max_sample_id + 1);
    compute_offsets(ctx);

    auto input_norm = norm_.forward(input);
    auto q = q_proj_.forward(input_norm);  // [batch, seq, hidden_dim]
    auto k = k_proj_.forward(input_norm);
    auto v = v_proj_.forward(input_norm);
    q.split_head(num_heads_).transpose(1, 2);  // [batch, num_head, seq, head_dim]
    k.split_head(num_heads_kv_).transpose(1, 2);  // [batch, num_head_kv, seq, head_dim]
    v.split_head(num_heads_kv_).transpose(1, 2);  // [batch, num_head_kv, seq, head_dim]
    if (use_qk_norm_) {
        q = q_norm_.forward(q);
        k = k_norm_.forward(k);
    }
    validate_tensor_size(q, "q");
    validate_tensor_size(k, "k");
    validate_tensor_size(v, "v");
    apply_rope_offsets(q, k, ctx);

    expand_kv_heads(k, v);
    validate_tensor_size(k, "k_repeat");
    validate_tensor_size(v, "v_repeat");
    append_kv_cache(k, v, ctx);
    auto cache_k_batch = build_active_cache(cache_k_by_sample_, ctx);
    auto cache_v_batch = build_active_cache(cache_v_by_sample_, ctx);
    if (cache_k_batch.valid_lens != cache_v_batch.valid_lens) {
        throw std::invalid_argument("forward_cpu: K/V active cache lengths mismatch.");
    }
    validate_tensor_size(cache_k_batch.cache, "cache_k_batch");
    validate_tensor_size(cache_v_batch.cache, "cache_v_batch");
    cache_k_batch.cache.transpose(2, 3);  // [batch, num_head, head_dim, seq]

    int seq_len = q.shape()[2];
    ctx.seq_len = seq_len;
    auto scores = ops::matmul_4d(q, cache_k_batch.cache);  // [batch, num_head, seq, seq]
    apply_attention_masks(scores, ctx, cache_k_batch.valid_lens);
    scores.scale_inplace(1.0f / std::sqrt(head_dim_));  // scale scores
    auto attention = scores.softmax();
    auto attn_output = ops::matmul_4d(attention, cache_v_batch.cache);  // [batch, num_head, seq, head_dim]
    const auto input_shape = input.shape();
    attn_output.transpose(1, 2).reshape({input_shape[0], input_shape[1], hidden_dim_});
    auto output = o_proj_.forward(attn_output);
    return output;
}

#ifdef USE_CUDA
Tensor SelfAttn::forward_cuda(const Tensor& input,
                              const std::vector<int>& sample_ids,
                              const std::vector<int>& offsets) {
    if (!cuda_state_) {
        cuda_state_ = std::make_unique<cuda::ops::SelfAttnCudaState>();
    }
    cuda::ops::SelfAttnCudaParams params{};
    params.hidden_dim = hidden_dim_;
    params.num_heads = num_heads_;
    params.num_heads_kv = num_heads_kv_;
    params.head_dim = head_dim_;
    params.rope_theta = rope_theta_;

    Tensor output = cuda::ops::self_attn_forward_cuda(
        input, sample_ids, offsets, pad_lens_by_sample_, params,
        norm_.weight(),
        q_proj_.weights(), q_proj_.bias(),
        k_proj_.weights(), k_proj_.bias(),
        v_proj_.weights(), v_proj_.bias(),
        o_proj_.weights(), o_proj_.bias(),
        *cuda_state_);

    for (int sample_id : sample_ids) {
        ensure_cache_capacity(sample_id + 1);
        cache_len_by_sample_[sample_id] = cuda_state_->cache_len(sample_id);
        cache_k_by_sample_[sample_id] = Tensor();
        cache_v_by_sample_[sample_id] = Tensor();
    }
    return output;
}
#endif

void SelfAttn::validate_forward_inputs(const ForwardContext& ctx) const {
    if (ctx.input.shape()[0] != static_cast<int>(ctx.sample_ids.size())) {
        throw std::invalid_argument("SelfAttn forward: input batch size must match sample_ids size.");
    }
}

void SelfAttn::compute_offsets(ForwardContext& ctx) const {
    ctx.offsets.clear();
    ctx.offsets.reserve(ctx.sample_ids.size());
    if (ctx.pos_offsets && ctx.pos_offsets->size() == ctx.sample_ids.size()) {
        ctx.offsets.assign(ctx.pos_offsets->begin(), ctx.pos_offsets->end());
        return;
    }
    for (int sample_id : ctx.sample_ids) {
        ctx.offsets.push_back(cache_len_by_sample_[sample_id]);
    }
}

void SelfAttn::apply_rope_offsets(Tensor& q, Tensor& k, const ForwardContext& ctx) {
    if (is_uniform_offset(ctx.offsets)) {
        int offset0 = ctx.offsets.empty() ? 0 : ctx.offsets[0];
        ops::apply_rope(q, offset0, rope_theta_);
        ops::apply_rope(k, offset0, rope_theta_);
        return;
    }
    ops::apply_rope(q, ctx.offsets, rope_theta_);
    ops::apply_rope(k, ctx.offsets, rope_theta_);
}

void SelfAttn::expand_kv_heads(Tensor& k, Tensor& v) {
    k.repeat(num_heads_ / num_heads_kv_, 1);  // [batch, num_head_kv, seq, head_dim]
    v.repeat(num_heads_ / num_heads_kv_, 1);  // [batch, num_head_kv, seq, head_dim]
}

void SelfAttn::apply_attention_masks(Tensor& scores,
                                     const ForwardContext& ctx,
                                     const std::vector<int>& valid_lens) const {
    int total_len = scores.shape()[3];
    if (ctx.seq_len > 1) {
        apply_causal_mask(scores, ctx.seq_len, total_len);
    }
    apply_valid_length_mask(scores, valid_lens);
    if (!pad_lens_by_sample_.empty()) {
        apply_padding_mask(scores, ctx.sample_ids, pad_lens_by_sample_);
    }
}

void SelfAttn::init_kv_cache(int batch_size) {
    cache_k_by_sample_.assign(batch_size, Tensor());
    cache_v_by_sample_.assign(batch_size, Tensor());
    cache_len_by_sample_.assign(batch_size, 0);
#ifdef USE_CUDA
    if (!cuda_state_) {
        cuda_state_ = std::make_unique<cuda::ops::SelfAttnCudaState>();
    }
    cuda_state_->init_kv_cache(batch_size);
    cuda_enabled_ = cuda_supported_;
#endif
}

void SelfAttn::clear_kv_cache(int sample_id) {
    if (sample_id < 0 || sample_id >= static_cast<int>(cache_len_by_sample_.size())) {
        return;
    }
    cache_k_by_sample_[sample_id] = Tensor();
    cache_v_by_sample_[sample_id] = Tensor();
    cache_len_by_sample_[sample_id] = 0;
#ifdef USE_CUDA
    if (cuda_state_) {
        cuda_state_->clear_kv_cache(sample_id);
    }
#endif
}

void SelfAttn::append_kv_cache(const Tensor& k, const Tensor& v, const ForwardContext& ctx) {
    int batch = k.shape()[0];
    if (batch != static_cast<int>(ctx.sample_ids.size())) {
        throw std::invalid_argument("append_kv_cache: k batch size mismatch with sample_ids.");
    }
    int seq_len = k.shape()[2];
    for (int i = 0; i < batch; ++i) {
        int sample_id = ctx.sample_ids[i];
        ensure_cache_capacity(sample_id + 1);
        Tensor k_slice = slice_batch_4d(k, i);
        Tensor v_slice = slice_batch_4d(v, i);
        if (cache_len_by_sample_[sample_id] == 0) {
            cache_k_by_sample_[sample_id] = std::move(k_slice);
            cache_v_by_sample_[sample_id] = std::move(v_slice);
        } else {
            cache_k_by_sample_[sample_id] = ops::concat({cache_k_by_sample_[sample_id], k_slice}, 2);
            cache_v_by_sample_[sample_id] = ops::concat({cache_v_by_sample_[sample_id], v_slice}, 2);
        }
        cache_len_by_sample_[sample_id] += seq_len;
    }
}

BatchedCacheView SelfAttn::build_active_cache(const std::vector<Tensor>& cache_by_sample,
                                              const ForwardContext& ctx) const {
    return build_padded_active_cache(cache_by_sample, ctx.sample_ids);
}

void SelfAttn::ensure_cache_capacity(int min_size) {
    if (min_size <= 0) {
        return;
    }
    if (static_cast<int>(cache_k_by_sample_.size()) >= min_size) {
        return;
    }
    cache_k_by_sample_.resize(min_size);
    cache_v_by_sample_.resize(min_size);
    cache_len_by_sample_.resize(min_size, 0);
}

void SelfAttn::reset_kv_cache() {
    cache_k_by_sample_.clear();
    cache_v_by_sample_.clear();
    cache_len_by_sample_.clear();
#ifdef USE_CUDA
    if (cuda_state_) {
        cuda_state_->reset_kv_cache();
    }
#endif
}

void SelfAttn::set_pad_lens(const std::vector<int>& pad_lens) {
    pad_lens_by_sample_ = pad_lens;
}

void SelfAttn::set_cuda_enabled(bool enabled) {
#ifdef USE_CUDA
    if (!enabled) {
        cuda_enabled_ = false;
        if (cuda_state_) {
            cuda_state_->reset_kv_cache();
        }
        return;
    }
    if (!cuda_supported_) {
        cuda_enabled_ = false;
        spdlog::warn("SelfAttn CUDA request ignored because this layer is CPU-only.");
        return;
    }
    cuda_enabled_ = true;
#else
    (void)enabled;
#endif
}

} // namespace easy_llm
