#ifndef EASY_LLM_CUDA_OPS_SELF_ATTN_HPP
#define EASY_LLM_CUDA_OPS_SELF_ATTN_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "tensor.hpp"

namespace easy_llm {
namespace cuda {
namespace ops {

struct SelfAttnCudaStats {
    std::uint64_t scratch_reallocations{0};
    std::uint64_t pad_lens_uploads{0};
    std::uint64_t decode_seq1_path_hits{0};
    std::uint64_t decode_fused_attention_hits{0};
    std::uint64_t prefill_fused_attention_hits{0};
    std::uint64_t decode_graph_captures{0};
    std::uint64_t decode_graph_launches{0};
    std::uint64_t repeat_head_materializations{0};
    std::uint64_t kv_cache_head_memcpy_ops{0};
};

struct SelfAttnCudaParams {
    int hidden_dim{0};
    int num_heads{0};
    int num_heads_kv{0};
    int head_dim{0};
    float rope_theta{10000.0f};
    bool use_qk_norm{false};
};

class SelfAttnCudaState {
public:
    struct Impl;

    SelfAttnCudaState();
    ~SelfAttnCudaState();
    SelfAttnCudaState(SelfAttnCudaState&& other) noexcept;
    SelfAttnCudaState& operator=(SelfAttnCudaState&& other) noexcept;
    SelfAttnCudaState(const SelfAttnCudaState&) = delete;
    SelfAttnCudaState& operator=(const SelfAttnCudaState&) = delete;

    void init_kv_cache(int batch_size);
    void clear_kv_cache(int sample_id);
    void reset_kv_cache();
    int cache_len(int sample_id) const;
    SelfAttnCudaStats stats() const;
    void reset_stats();

private:
    std::unique_ptr<Impl> impl_;

    friend Tensor self_attn_forward_cuda(
        const Tensor& input,
        const std::vector<int>& sample_ids,
        const std::vector<int>& offsets,
        const std::vector<int>& pad_lens_by_sample,
        const SelfAttnCudaParams& params,
        const Tensor& norm_weight,
        const Tensor& q_norm_weight,
        const Tensor& k_norm_weight,
        const Tensor& q_weight, const Tensor& q_bias,
        const Tensor& k_weight, const Tensor& k_bias,
        const Tensor& v_weight, const Tensor& v_bias,
        const Tensor& o_weight, const Tensor& o_bias,
        SelfAttnCudaState& state);
};

Tensor self_attn_forward_cuda(
    const Tensor& input,
    const std::vector<int>& sample_ids,
    const std::vector<int>& offsets,
    const std::vector<int>& pad_lens_by_sample,
    const SelfAttnCudaParams& params,
    const Tensor& norm_weight,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& q_weight, const Tensor& q_bias,
    const Tensor& k_weight, const Tensor& k_bias,
    const Tensor& v_weight, const Tensor& v_bias,
    const Tensor& o_weight, const Tensor& o_bias,
    SelfAttnCudaState& state);

} // namespace ops
} // namespace cuda
} // namespace easy_llm

#endif // EASY_LLM_CUDA_OPS_SELF_ATTN_HPP
