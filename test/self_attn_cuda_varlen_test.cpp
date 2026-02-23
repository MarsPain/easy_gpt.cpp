#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "cuda/ops/self_attn.hpp"
#include "cuda/runtime.hpp"
#include "models/cache_batching.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace easy_llm {
namespace {

Tensor make_tensor(std::mt19937& rng, const std::vector<int>& shape, float scale = 0.1f) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<data_type> data;
    data.reserve(size);
    for (int i = 0; i < size; ++i) {
        data.emplace_back(data_type(dist(rng)));
    }
    return Tensor(data, shape);
}

Tensor make_zero_tensor(const std::vector<int>& shape) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    std::vector<data_type> data(static_cast<size_t>(size), data_type(0.0f));
    return Tensor(data, shape);
}

Tensor add_bias(const Tensor& input, const Tensor& bias) {
    Tensor out = input;
    if (out.shape().size() != 3 || bias.shape().size() != 1 || bias.size() != out.shape()[2]) {
        return out;
    }
    const int batch = out.shape()[0];
    const int seq_len = out.shape()[1];
    const int hidden = out.shape()[2];
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden; ++h) {
                const int idx = (b * seq_len + s) * hidden + h;
                out[idx] = out[idx] + bias[h];
            }
        }
    }
    return out;
}

Tensor rms_norm_ref(const Tensor& input, const Tensor& weight) {
    if (input.shape().empty() || input.shape().back() != weight.size()) {
        throw std::invalid_argument("rms_norm_ref: invalid shape.");
    }
    Tensor out = input;
    const int features = input.shape().back();
    const int vectors = input.size() / features;
    for (int v = 0; v < vectors; ++v) {
        float sum_sq = 0.0f;
        const int base = v * features;
        for (int i = 0; i < features; ++i) {
            const float val = static_cast<float>(input[base + i]);
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(features) + 1e-6f);
        for (int i = 0; i < features; ++i) {
            out[base + i] = data_type(static_cast<float>(weight[i]) * static_cast<float>(input[base + i]) * inv_rms);
        }
    }
    return out;
}

Tensor slice_batch_4d(const Tensor& input, int batch_index) {
    const auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("slice_batch_4d expects 4D tensor.");
    }
    const int block = shape[1] * shape[2] * shape[3];
    std::vector<data_type> data(block);
    const int start = batch_index * block;
    std::copy(input.data().begin() + start, input.data().begin() + start + block, data.begin());
    return Tensor(data, {1, shape[1], shape[2], shape[3]});
}

void write_batch_4d(Tensor& output, int batch_index, const Tensor& slice) {
    const auto shape = output.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("write_batch_4d expects 4D tensor.");
    }
    const int block = shape[1] * shape[2] * shape[3];
    const int start = batch_index * block;
    std::copy(slice.data().begin(), slice.data().end(), output.data().begin() + start);
}

void apply_causal_mask(Tensor& scores, int seq_len, int total_len) {
    const int batch = scores.shape()[0];
    const int heads = scores.shape()[1];
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();
    const int start_pos = total_len - seq_len;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                const int mask_start = start_pos + s + 1;
                if (mask_start >= total_len) {
                    continue;
                }
                const int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                for (int k = mask_start; k < total_len; ++k) {
                    data[row_offset + k] = data_type(neg_inf);
                }
            }
        }
    }
}

void apply_padding_mask(Tensor& scores, const std::vector<int>& sample_ids, const std::vector<int>& pad_lens) {
    const int batch = scores.shape()[0];
    const int heads = scores.shape()[1];
    const int seq_len = scores.shape()[2];
    const int total_len = scores.shape()[3];
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();
    for (int b = 0; b < batch; ++b) {
        const int sample_id = sample_ids[b];
        if (sample_id < 0 || sample_id >= static_cast<int>(pad_lens.size())) {
            continue;
        }
        const int pad_len = std::clamp(pad_lens[sample_id], 0, total_len);
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                const int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                for (int k = 0; k < pad_len; ++k) {
                    data[row_offset + k] = data_type(neg_inf);
                }
            }
        }
    }
}

Tensor run_cpu_reference_varlen(
    const Tensor& input,
    const std::vector<int>& sample_ids,
    const std::vector<int>& offsets,
    const std::vector<int>& pad_lens,
    const cuda::ops::SelfAttnCudaParams& params,
    const Tensor& norm_weight, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
    const Tensor& q_weight, const Tensor& q_bias,
    const Tensor& k_weight, const Tensor& k_bias,
    const Tensor& v_weight, const Tensor& v_bias,
    const Tensor& o_weight, const Tensor& o_bias,
    std::vector<Tensor>& cache_k_by_sample,
    std::vector<Tensor>& cache_v_by_sample,
    std::vector<int>& cache_len_by_sample) {
    Tensor input_norm = rms_norm_ref(input, norm_weight);
    Tensor q = add_bias(ops::matmul_3d(input_norm, q_weight), q_bias);
    Tensor k = add_bias(ops::matmul_3d(input_norm, k_weight), k_bias);
    Tensor v = add_bias(ops::matmul_3d(input_norm, v_weight), v_bias);

    q.split_head(params.num_heads).transpose(1, 2);
    k.split_head(params.num_heads_kv).transpose(1, 2);
    v.split_head(params.num_heads_kv).transpose(1, 2);
    if (params.use_qk_norm) {
        q = rms_norm_ref(q, q_norm_weight);
        k = rms_norm_ref(k, k_norm_weight);
    }

    bool uniform_offset = true;
    for (int i = 1; i < static_cast<int>(offsets.size()); ++i) {
        if (offsets[i] != offsets[0]) {
            uniform_offset = false;
            break;
        }
    }
    if (uniform_offset) {
        const int offset = offsets.empty() ? 0 : offsets[0];
        ops::apply_rope(q, offset, params.rope_theta);
        ops::apply_rope(k, offset, params.rope_theta);
    } else {
        for (int i = 0; i < static_cast<int>(sample_ids.size()); ++i) {
            Tensor q_slice = slice_batch_4d(q, i);
            Tensor k_slice = slice_batch_4d(k, i);
            ops::apply_rope(q_slice, offsets[i], params.rope_theta);
            ops::apply_rope(k_slice, offsets[i], params.rope_theta);
            write_batch_4d(q, i, q_slice);
            write_batch_4d(k, i, k_slice);
        }
    }

    const int repeat_factor = params.num_heads / params.num_heads_kv;
    k.repeat(repeat_factor, 1);
    v.repeat(repeat_factor, 1);

    const int batch = static_cast<int>(sample_ids.size());
    const int seq_len = input.shape()[1];
    for (int i = 0; i < batch; ++i) {
        const int sample_id = sample_ids[i];
        Tensor k_slice = slice_batch_4d(k, i);
        Tensor v_slice = slice_batch_4d(v, i);
        if (cache_len_by_sample[sample_id] == 0) {
            cache_k_by_sample[sample_id] = std::move(k_slice);
            cache_v_by_sample[sample_id] = std::move(v_slice);
        } else {
            cache_k_by_sample[sample_id] = ops::concat({cache_k_by_sample[sample_id], k_slice}, 2);
            cache_v_by_sample[sample_id] = ops::concat({cache_v_by_sample[sample_id], v_slice}, 2);
        }
        cache_len_by_sample[sample_id] += seq_len;
    }

    BatchedCacheView cache_k_batch = build_padded_active_cache(cache_k_by_sample, sample_ids);
    BatchedCacheView cache_v_batch = build_padded_active_cache(cache_v_by_sample, sample_ids);
    if (cache_k_batch.valid_lens != cache_v_batch.valid_lens) {
        throw std::invalid_argument("run_cpu_reference_varlen: K/V valid_lens mismatch.");
    }
    std::vector<int> valid_lens = cache_k_batch.valid_lens;
    cache_k_batch.cache.transpose(2, 3);

    Tensor scores = ops::matmul_4d(q, cache_k_batch.cache);
    const int total_len = scores.shape()[3];
    if (seq_len > 1) {
        apply_causal_mask(scores, seq_len, total_len);
    }
    apply_valid_length_mask(scores, valid_lens);
    if (!pad_lens.empty()) {
        apply_padding_mask(scores, sample_ids, pad_lens);
    }
    scores.scale_inplace(1.0f / std::sqrt(static_cast<float>(params.head_dim)));

    Tensor attn = scores.softmax();
    Tensor out = ops::matmul_4d(attn, cache_v_batch.cache);
    out.transpose(1, 2).reshape(input.shape());
    return add_bias(ops::matmul_3d(out, o_weight), o_bias);
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("max_abs_diff: shape mismatch.");
    }
    float max_v = 0.0f;
    for (int i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (diff > max_v) {
            max_v = diff;
        }
    }
    return max_v;
}

} // namespace
} // namespace easy_llm

int main() {
    using namespace easy_llm;

    if (!cuda::initialize()) {
        std::cout << "SKIP: CUDA unavailable\n";
        return 0;
    }

    std::mt19937 rng(123);
    cuda::ops::SelfAttnCudaParams params{};
    params.hidden_dim = 8;
    params.num_heads = 2;
    params.num_heads_kv = 1;
    params.head_dim = 6;
    params.rope_theta = 10000.0f;
    params.use_qk_norm = true;

    const int slot_count = 2;
    const std::vector<int> pad_lens{1, 0};
    const float tolerance = 9e-2f;

    Tensor norm_weight = make_tensor(rng, {params.hidden_dim});
    Tensor q_norm_weight = make_tensor(rng, {params.head_dim});
    Tensor k_norm_weight = make_tensor(rng, {params.head_dim});
    Tensor q_weight = make_tensor(rng, {params.num_heads * params.head_dim, params.hidden_dim});
    Tensor q_bias = make_zero_tensor({params.num_heads * params.head_dim});
    Tensor k_weight = make_tensor(rng, {params.num_heads_kv * params.head_dim, params.hidden_dim});
    Tensor k_bias = make_zero_tensor({params.num_heads_kv * params.head_dim});
    Tensor v_weight = make_tensor(rng, {params.num_heads_kv * params.head_dim, params.hidden_dim});
    Tensor v_bias = make_zero_tensor({params.num_heads_kv * params.head_dim});
    Tensor o_weight = make_tensor(rng, {params.hidden_dim, params.num_heads * params.head_dim});
    Tensor o_bias = make_tensor(rng, {params.hidden_dim});

    std::vector<Tensor> cpu_cache_k(slot_count);
    std::vector<Tensor> cpu_cache_v(slot_count);
    std::vector<int> cpu_cache_len(slot_count, 0);

    cuda::ops::SelfAttnCudaState cuda_state;
    cuda_state.init_kv_cache(slot_count);

    // Step 1: prefill two samples together.
    const std::vector<int> prefill_sample_ids{0, 1};
    const std::vector<int> prefill_offsets{-1, 0};
    Tensor prefill_input = make_tensor(rng, {2, 3, params.hidden_dim});
    Tensor cpu_prefill = run_cpu_reference_varlen(prefill_input,
                                                  prefill_sample_ids,
                                                  prefill_offsets,
                                                  pad_lens,
                                                  params,
                                                  norm_weight,
                                                  q_norm_weight, k_norm_weight,
                                                  q_weight, q_bias,
                                                  k_weight, k_bias,
                                                  v_weight, v_bias,
                                                  o_weight, o_bias,
                                                  cpu_cache_k, cpu_cache_v, cpu_cache_len);
    Tensor cuda_prefill = cuda::ops::self_attn_forward_cuda(prefill_input,
                                                            prefill_sample_ids,
                                                            prefill_offsets,
                                                            pad_lens,
                                                            params,
                                                            norm_weight,
                                                            q_norm_weight, k_norm_weight,
                                                            q_weight, q_bias,
                                                            k_weight, k_bias,
                                                            v_weight, v_bias,
                                                            o_weight, o_bias,
                                                            cuda_state);
    if (max_abs_diff(cpu_prefill, cuda_prefill) > tolerance) {
        std::cerr << "FAIL: prefill parity check failed\n";
        return 1;
    }

    // Step 2: decode only sample 0 once to create mixed cache lengths.
    const std::vector<int> single_sample_ids{0};
    const std::vector<int> single_offsets{cpu_cache_len[0]};
    Tensor decode_input_single = make_tensor(rng, {1, 1, params.hidden_dim});
    Tensor cpu_decode_single = run_cpu_reference_varlen(decode_input_single,
                                                        single_sample_ids,
                                                        single_offsets,
                                                        pad_lens,
                                                        params,
                                                        norm_weight,
                                                        q_norm_weight, k_norm_weight,
                                                        q_weight, q_bias,
                                                        k_weight, k_bias,
                                                        v_weight, v_bias,
                                                        o_weight, o_bias,
                                                        cpu_cache_k, cpu_cache_v, cpu_cache_len);
    Tensor cuda_decode_single = cuda::ops::self_attn_forward_cuda(decode_input_single,
                                                                  single_sample_ids,
                                                                  single_offsets,
                                                                  pad_lens,
                                                                  params,
                                                                  norm_weight,
                                                                  q_norm_weight, k_norm_weight,
                                                                  q_weight, q_bias,
                                                                  k_weight, k_bias,
                                                                  v_weight, v_bias,
                                                                  o_weight, o_bias,
                                                                  cuda_state);
    if (max_abs_diff(cpu_decode_single, cuda_decode_single) > tolerance) {
        std::cerr << "FAIL: single-sample decode parity check failed\n";
        return 1;
    }
    if (cpu_cache_len[0] == cpu_cache_len[1]) {
        std::cerr << "FAIL: setup error, cache lengths should diverge before varlen decode\n";
        return 1;
    }

    // Step 3: decode both samples together with mixed active cache lengths.
    const std::vector<int> mixed_sample_ids{0, 1};
    const std::vector<int> mixed_offsets{cpu_cache_len[0], cpu_cache_len[1]};
    Tensor decode_input_mixed = make_tensor(rng, {2, 1, params.hidden_dim});
    Tensor cpu_decode_mixed = run_cpu_reference_varlen(decode_input_mixed,
                                                       mixed_sample_ids,
                                                       mixed_offsets,
                                                       pad_lens,
                                                       params,
                                                       norm_weight,
                                                       q_norm_weight, k_norm_weight,
                                                       q_weight, q_bias,
                                                       k_weight, k_bias,
                                                       v_weight, v_bias,
                                                       o_weight, o_bias,
                                                       cpu_cache_k, cpu_cache_v, cpu_cache_len);
    Tensor cuda_decode_mixed;
    try {
        cuda_decode_mixed = cuda::ops::self_attn_forward_cuda(decode_input_mixed,
                                                              mixed_sample_ids,
                                                              mixed_offsets,
                                                              pad_lens,
                                                              params,
                                                              norm_weight,
                                                              q_norm_weight, k_norm_weight,
                                                              q_weight, q_bias,
                                                              k_weight, k_bias,
                                                              v_weight, v_bias,
                                                              o_weight, o_bias,
                                                              cuda_state);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: mixed-length decode should be supported, got exception: " << e.what() << "\n";
        return 1;
    }
    const float mixed_diff = max_abs_diff(cpu_decode_mixed, cuda_decode_mixed);
    std::cout << "mixed_decode_max_abs_diff=" << mixed_diff << "\n";
    if (mixed_diff > tolerance) {
        std::cerr << "FAIL: mixed-length decode parity check failed\n";
        return 1;
    }

    if (cuda_state.cache_len(0) != cpu_cache_len[0] || cuda_state.cache_len(1) != cpu_cache_len[1]) {
        std::cerr << "FAIL: CUDA cache lengths drift from CPU reference state\n";
        return 1;
    }
    if (cuda_state.cache_len(0) <= cuda_state.cache_len(1)) {
        std::cerr << "FAIL: expected sample 0 to remain longer after mixed decode setup\n";
        return 1;
    }

    std::cout << "PASS: self_attn CUDA varlen active cache decode\n";
    return 0;
}
