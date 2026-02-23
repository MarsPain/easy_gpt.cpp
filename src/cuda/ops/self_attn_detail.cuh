#ifndef EASY_LLM_CUDA_OPS_SELF_ATTN_DETAIL_CUH
#define EASY_LLM_CUDA_OPS_SELF_ATTN_DETAIL_CUH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_bf16.h>

#include "cuda/ops/self_attn.hpp"
#include "cuda/precision.hpp"
#include "spdlog/spdlog.h"

#include "../device_buffer.hpp"
#include "../runtime_internal.hpp"

namespace easy_llm {
namespace cuda {
namespace ops {

struct SampleCache {
    DeviceBuffer k;
    DeviceBuffer v;
    int len{0};
    int capacity_len{0};
};

struct TensorUploadCache {
    DeviceBuffer device;
    const void* host_ptr{nullptr};
    size_t bytes{0};
};

struct IntVectorUploadCache {
    DeviceBuffer device;
    const int* host_ptr{nullptr};
    size_t count{0};
    std::uint64_t checksum{0};
};

struct ForwardScratchBuffers {
    DeviceBuffer input;
    DeviceBuffer norm_out;
    DeviceBuffer q_proj;
    DeviceBuffer k_proj;
    DeviceBuffer v_proj;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer offsets;
    DeviceBuffer cache_k_batch;
    DeviceBuffer cache_v_batch;
    DeviceBuffer scores;
    DeviceBuffer sample_ids;
    DeviceBuffer context;
    DeviceBuffer merged;
    DeviceBuffer output;
    DeviceBuffer decode_active_len;
};

struct DecodeGraphCache {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};
    int batch{0};
    int sample_id{-1};
    int num_heads{0};
    int head_dim{0};
    int hidden_dim{0};
    int q_out_dim{0};
    int k_out_dim{0};
    int v_out_dim{0};
    int o_out_dim{0};
    int repeat_factor{0};
    int score_capacity{0};
    int pad_size{0};
    bool use_qk_norm{false};
    const void* cache_k_ptr{nullptr};
    const void* cache_v_ptr{nullptr};
    const int* d_pad_ptr{nullptr};
    const void* d_norm_weight_ptr{nullptr};
    const void* d_q_norm_weight_ptr{nullptr};
    const void* d_k_norm_weight_ptr{nullptr};
    const void* q_weight_ptr{nullptr};
    const void* k_weight_ptr{nullptr};
    const void* v_weight_ptr{nullptr};
    const void* o_weight_ptr{nullptr};
    const void* d_q_bias_ptr{nullptr};
    const void* d_k_bias_ptr{nullptr};
    const void* d_v_bias_ptr{nullptr};
    const void* d_o_bias_ptr{nullptr};

    void reset() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        batch = 0;
        sample_id = -1;
        num_heads = 0;
        head_dim = 0;
        hidden_dim = 0;
        q_out_dim = 0;
        k_out_dim = 0;
        v_out_dim = 0;
        o_out_dim = 0;
        repeat_factor = 0;
        score_capacity = 0;
        pad_size = 0;
        use_qk_norm = false;
        cache_k_ptr = nullptr;
        cache_v_ptr = nullptr;
        d_pad_ptr = nullptr;
        d_norm_weight_ptr = nullptr;
        d_q_norm_weight_ptr = nullptr;
        d_k_norm_weight_ptr = nullptr;
        q_weight_ptr = nullptr;
        k_weight_ptr = nullptr;
        v_weight_ptr = nullptr;
        o_weight_ptr = nullptr;
        d_q_bias_ptr = nullptr;
        d_k_bias_ptr = nullptr;
        d_v_bias_ptr = nullptr;
        d_o_bias_ptr = nullptr;
    }

    ~DecodeGraphCache() {
        reset();
    }
};

struct SelfAttnCudaState::Impl {
    std::vector<SampleCache> samples;
    int num_heads{0};
    int head_dim{0};
    DeviceBuffer rope_inv_freq;
    int rope_inv_freq_half_dim{0};
    float rope_inv_freq_theta{0.0f};
    ForwardScratchBuffers scratch;
    TensorUploadCache norm_weight_cache;
    TensorUploadCache q_norm_weight_cache;
    TensorUploadCache k_norm_weight_cache;
    TensorUploadCache q_bias_cache;
    TensorUploadCache k_bias_cache;
    TensorUploadCache v_bias_cache;
    TensorUploadCache o_bias_cache;
    IntVectorUploadCache pad_lens_cache;
    DecodeGraphCache decode_graph;
    SelfAttnCudaStats stats;
};

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T value);

#if defined(USE_FP32)
template <>
__device__ __forceinline__ float to_float<float>(float value) {
    return value;
}
#endif

#if defined(USE_BF16)
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
#endif

template <typename T>
__device__ __forceinline__ T from_float(float value);

#if defined(USE_FP32)
template <>
__device__ __forceinline__ float from_float<float>(float value) {
    return value;
}
#endif

#if defined(USE_BF16)
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
#endif

__device__ __forceinline__ int clamp_int(int value, int min_value, int max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

__device__ __forceinline__ int resolve_active_len(const int* active_len_ptr, int max_len, int batch_index) {
    int active_len = max_len;
    if (active_len_ptr != nullptr) {
        active_len = active_len_ptr[batch_index];
    }
    return clamp_int(active_len, 0, max_len);
}

__device__ __forceinline__ int resolve_pad_len(const int* sample_ids,
                                               const int* pad_lens,
                                               int pad_lens_size,
                                               int batch_index,
                                               int max_len) {
    if (sample_ids == nullptr || pad_lens == nullptr || pad_lens_size <= 0) {
        return 0;
    }
    int sample_id = sample_ids[batch_index];
    if (sample_id < 0 || sample_id >= pad_lens_size) {
        return 0;
    }
    return clamp_int(pad_lens[sample_id], 0, max_len);
}

__device__ __forceinline__ float block_reduce_max(float* shared, float value) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float* shared, float value) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__device__ __forceinline__ float load_scaled_masked_score(const T* row_scores,
                                                          int score_index,
                                                          int active_len,
                                                          int pad_len,
                                                          bool use_causal,
                                                          int causal_mask_start,
                                                          float scale) {
    float value = to_float(row_scores[score_index]);
    if (score_index >= active_len ||
        (pad_len > 0 && score_index < pad_len) ||
        (use_causal && score_index >= causal_mask_start)) {
        value = -INFINITY;
    }
    return value * scale;
}

template <typename T>
__device__ __forceinline__ void softmax_row_inplace(T* row_scores,
                                                     int score_len,
                                                     int active_len,
                                                     int pad_len,
                                                     bool use_causal,
                                                     int causal_mask_start,
                                                     float scale,
                                                     float* shared) {
    int tid = threadIdx.x;
    float local_max = -INFINITY;
    for (int k = tid; k < score_len; k += blockDim.x) {
        float v = load_scaled_masked_score(row_scores, k, active_len, pad_len, use_causal, causal_mask_start, scale);
        local_max = fmaxf(local_max, v);
    }
    float row_max = block_reduce_max(shared, local_max);

    float local_sum = 0.0f;
    for (int k = tid; k < score_len; k += blockDim.x) {
        float v = load_scaled_masked_score(row_scores, k, active_len, pad_len, use_causal, causal_mask_start, scale);
        float exp_v = 0.0f;
        if (isfinite(row_max)) {
            exp_v = expf(v - row_max);
        }
        row_scores[k] = from_float<T>(exp_v);
        local_sum += exp_v;
    }
    float denom = fmaxf(block_reduce_sum(shared, local_sum), FLT_EPSILON * 10.0f);
    for (int k = tid; k < score_len; k += blockDim.x) {
        float p = to_float(row_scores[k]) / denom;
        row_scores[k] = from_float<T>(p);
    }
}

template <typename T>
__device__ __forceinline__ void run_online_attention_row(const T* q,
                                                         const T* cache_k_interleaved,
                                                         const T* cache_v_interleaved,
                                                         T* context,
                                                         int batch_index,
                                                         int kv_head_index,
                                                         int q_base,
                                                         int out_base,
                                                         int num_heads_kv,
                                                         int cache_len_stride,
                                                         int key_limit,
                                                         int pad_len,
                                                         int head_dim,
                                                         float scale,
                                                         float* reduce,
                                                         float* q_shared,
                                                         float* acc_shared,
                                                         float* scalars) {
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = to_float(q[q_base + d]);
        acc_shared[d] = 0.0f;
    }
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (int t = 0; t < key_limit; ++t) {
        if (t < pad_len) {
            continue;
        }

        int cache_base = (((batch_index * cache_len_stride) + t) * num_heads_kv + kv_head_index) * head_dim;
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial += q_shared[d] * to_float(cache_k_interleaved[cache_base + d]);
        }
        reduce[tid] = partial;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float score = reduce[0] * scale;
            float next_max = fmaxf(running_max, score);
            float alpha = 0.0f;
            if (isfinite(running_max)) {
                alpha = expf(running_max - next_max);
            }
            float beta = expf(score - next_max);
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
            scalars[0] = alpha;
            scalars[1] = beta;
        }
        __syncthreads();

        float alpha = scalars[0];
        float beta = scalars[1];
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float v = to_float(cache_v_interleaved[cache_base + d]);
            acc_shared[d] = acc_shared[d] * alpha + v * beta;
        }
        __syncthreads();
    }

    if (tid == 0) {
        scalars[0] = running_sum;
    }
    __syncthreads();

    float inv_denom = 0.0f;
    if (scalars[0] > FLT_EPSILON * 10.0f) {
        inv_denom = 1.0f / scalars[0];
    }
    for (int d = tid; d < head_dim; d += blockDim.x) {
        context[out_base + d] = from_float<T>(acc_shared[d] * inv_denom);
    }
}

template <typename T>
__global__ void rms_norm_kernel(const T* input,
                                const T* weight,
                                T* output,
                                int num_vectors,
                                int hidden_dim,
                                float epsilon) {
    extern __shared__ float shared[];
    int vec = blockIdx.x;
    if (vec >= num_vectors) {
        return;
    }
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    int base = vec * hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = to_float(input[base + i]);
        local_sum += v * v;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_dim) + epsilon);
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[base + i]);
        float w = to_float(weight[i]);
        output[base + i] = from_float<T>(x * w * inv_rms);
    }
}

template <typename T>
__global__ void add_bias_kernel(T* data, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }
    int col = idx % cols;
    float v = to_float(data[idx]) + to_float(bias[col]);
    data[idx] = from_float<T>(v);
}

template <typename T>
__global__ void split_transpose_kernel(const T* input,
                                       T* output,
                                       int batch,
                                       int seq_len,
                                       int num_heads,
                                       int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    int hidden_dim = num_heads * head_dim;
    int in_idx = ((b * seq_len + s) * hidden_dim) + (h * head_dim + d);
    output[idx] = input[in_idx];
}

template <typename T>
__global__ void split_transpose_seq1_kernel(const T* input,
                                            T* output,
                                            int batch,
                                            int num_heads,
                                            int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int h = (idx / head_dim) % num_heads;
    int b = idx / (head_dim * num_heads);
    int hidden_dim = num_heads * head_dim;
    int in_idx = b * hidden_dim + h * head_dim + d;
    output[idx] = input[in_idx];
}

template <typename T>
__global__ void append_kv_interleaved_kernel(const T* src_heads_major,
                                             T* dst_interleaved,
                                             int old_len,
                                             int seq_len,
                                             int num_heads_kv,
                                             int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads_kv * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = idx / (head_dim * seq_len);
    int src_idx = ((h * seq_len + s) * head_dim) + d;
    int dst_idx = (((old_len + s) * num_heads_kv + h) * head_dim) + d;
    dst_interleaved[dst_idx] = src_heads_major[src_idx];
}

template <typename T>
__global__ void pack_cache_interleaved_to_heads_kernel(const T* src_interleaved,
                                                       T* dst_heads_major,
                                                       int src_len,
                                                       int dst_len,
                                                       int num_heads_kv,
                                                       int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = src_len * num_heads_kv * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int t = (idx / head_dim) % src_len;
    int h = idx / (head_dim * src_len);
    int src_idx = ((t * num_heads_kv + h) * head_dim) + d;
    int dst_idx = ((h * dst_len + t) * head_dim) + d;
    dst_heads_major[dst_idx] = src_interleaved[src_idx];
}

template <typename T>
__global__ void rope_kernel(T* input,
                            const int* offsets,
                            const float* inv_freq,
                            int batch,
                            int num_heads,
                            int seq_len,
                            int head_dim) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * half_dim;
    if (idx >= total) {
        return;
    }
    int k = idx % half_dim;
    int s = (idx / half_dim) % seq_len;
    int h = (idx / (half_dim * seq_len)) % num_heads;
    int b = idx / (half_dim * seq_len * num_heads);

    int base = ((b * num_heads + h) * seq_len + s) * head_dim;
    int idx1 = base + k;
    int idx2 = idx1 + half_dim;
    float x1 = to_float(input[idx1]);
    float x2 = to_float(input[idx2]);
    float angle = static_cast<float>(s + offsets[b]) * inv_freq[k];
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    input[idx1] = from_float<T>(x1 * cos_v - x2 * sin_v);
    input[idx2] = from_float<T>(x1 * sin_v + x2 * cos_v);
}

template <typename T>
__global__ void rope_seq1_kernel(T* input,
                                 const int* offsets,
                                 const float* inv_freq,
                                 int batch,
                                 int num_heads,
                                 int head_dim) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * half_dim;
    if (idx >= total) {
        return;
    }
    int k = idx % half_dim;
    int h = (idx / half_dim) % num_heads;
    int b = idx / (half_dim * num_heads);

    int base = (b * num_heads + h) * head_dim;
    int idx1 = base + k;
    int idx2 = idx1 + half_dim;
    float x1 = to_float(input[idx1]);
    float x2 = to_float(input[idx2]);
    float angle = static_cast<float>(offsets[b]) * inv_freq[k];
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    input[idx1] = from_float<T>(x1 * cos_v - x2 * sin_v);
    input[idx2] = from_float<T>(x1 * sin_v + x2 * cos_v);
}

template <typename T>
__global__ void mask_scale_softmax_kernel(T* scores,
                                          const int* sample_ids,
                                          const int* pad_lens,
                                          int pad_lens_size,
                                          int batch,
                                          int heads,
                                          int seq_len,
                                          int total_len,
                                          float scale) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int row_count = batch * heads * seq_len;
    if (row >= row_count) {
        return;
    }
    int b = row / (heads * seq_len);
    int rem = row % (heads * seq_len);
    int s = rem % seq_len;
    int row_base = row * total_len;
    int active_len = total_len;
    int mask_start = (total_len - seq_len) + s + 1;
    bool use_causal = seq_len > 1;
    int pad_len = resolve_pad_len(sample_ids, pad_lens, pad_lens_size, b, active_len);

    softmax_row_inplace(scores + row_base,
                        total_len,
                        active_len,
                        pad_len,
                        use_causal,
                        mask_start,
                        scale,
                        shared);
}

template <typename T>
__global__ void mask_scale_softmax_decode_kernel(T* scores,
                                                 const int* sample_ids,
                                                 const int* pad_lens,
                                                 int pad_lens_size,
                                                 int batch,
                                                 int heads,
                                                 int score_len,
                                                 const int* active_len_ptr,
                                                 float scale) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int row_count = batch * heads;
    if (row >= row_count) {
        return;
    }

    int b = row / heads;
    int row_base = row * score_len;
    int active_len = resolve_active_len(active_len_ptr, score_len, b);
    int pad_len = resolve_pad_len(sample_ids, pad_lens, pad_lens_size, b, active_len);

    softmax_row_inplace(scores + row_base,
                        score_len,
                        active_len,
                        pad_len,
                        false,
                        score_len,
                        scale,
                        shared);
}

template <typename T>
__global__ void fused_decode_attention_kernel(const T* q,
                                              const T* cache_k_interleaved,
                                              const T* cache_v_interleaved,
                                              T* context,
                                              const int* sample_ids,
                                              const int* pad_lens,
                                              int pad_lens_size,
                                              int batch,
                                              int num_heads,
                                              int num_heads_kv,
                                              int repeat_factor,
                                              int head_dim,
                                              int cache_capacity_len,
                                              const int* active_len_ptr,
                                              float scale) {
    extern __shared__ float shared[];
    float* reduce = shared;
    float* q_shared = reduce + blockDim.x;
    float* acc_shared = q_shared + head_dim;
    float* scalars = acc_shared + head_dim;

    int row = blockIdx.x;
    int row_count = batch * num_heads;
    if (row >= row_count) {
        return;
    }

    int b = row / num_heads;
    int h = row % num_heads;
    int kv_h = h / repeat_factor;
    int q_base = (b * num_heads + h) * head_dim;
    int active_len = resolve_active_len(active_len_ptr, cache_capacity_len, b);
    int pad_len = resolve_pad_len(sample_ids, pad_lens, pad_lens_size, b, active_len);
    int out_base = (b * num_heads + h) * head_dim;

    run_online_attention_row(q,
                             cache_k_interleaved,
                             cache_v_interleaved,
                             context,
                             b,
                             kv_h,
                             q_base,
                             out_base,
                             num_heads_kv,
                             cache_capacity_len,
                             active_len,
                             pad_len,
                             head_dim,
                             scale,
                             reduce,
                             q_shared,
                             acc_shared,
                             scalars);
}

template <typename T>
__global__ void fused_prefill_attention_kernel(const T* q,
                                               const T* cache_k_interleaved,
                                               const T* cache_v_interleaved,
                                               T* context,
                                               const int* sample_ids,
                                               const int* pad_lens,
                                               int pad_lens_size,
                                               int batch,
                                               int num_heads,
                                               int num_heads_kv,
                                               int repeat_factor,
                                               int seq_len,
                                               int total_len,
                                               int head_dim,
                                               float scale) {
    extern __shared__ float shared[];
    float* reduce = shared;
    float* q_shared = reduce + blockDim.x;
    float* acc_shared = q_shared + head_dim;
    float* scalars = acc_shared + head_dim;

    int row = blockIdx.x;
    int row_count = batch * num_heads * seq_len;
    if (row >= row_count) {
        return;
    }

    int b = row / (num_heads * seq_len);
    int rem = row % (num_heads * seq_len);
    int h = rem / seq_len;
    int s = rem % seq_len;
    int kv_h = h / repeat_factor;
    int q_base = ((b * num_heads + h) * seq_len + s) * head_dim;
    int key_limit = clamp_int(total_len - seq_len + s + 1, 0, total_len);
    int pad_len = resolve_pad_len(sample_ids, pad_lens, pad_lens_size, b, key_limit);
    int out_base = ((b * num_heads + h) * seq_len + s) * head_dim;

    run_online_attention_row(q,
                             cache_k_interleaved,
                             cache_v_interleaved,
                             context,
                             b,
                             kv_h,
                             q_base,
                             out_base,
                             num_heads_kv,
                             total_len,
                             key_limit,
                             pad_len,
                             head_dim,
                             scale,
                             reduce,
                             q_shared,
                             acc_shared,
                             scalars);
}

template <typename T>
__global__ void merge_heads_kernel(const T* input,
                                   T* output,
                                   int batch,
                                   int seq_len,
                                   int num_heads,
                                   int head_dim) {
    int hidden_dim = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_dim;
    if (idx >= total) {
        return;
    }
    int hidden_idx = idx % hidden_dim;
    int d = hidden_idx % head_dim;
    int h = hidden_idx / head_dim;
    int s = (idx / hidden_dim) % seq_len;
    int b = idx / (hidden_dim * seq_len);
    int in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    output[idx] = input[in_idx];
}

void validate_rank(const Tensor& tensor, int rank, const char* name) {
    if (static_cast<int>(tensor.shape().size()) != rank) {
        throw std::invalid_argument(std::string(name) + " rank mismatch.");
    }
}

void validate_vector_like(const Tensor& tensor, const char* name) {
    const auto shape = tensor.shape();
    if (shape.size() == 1) {
        if (tensor.size() != shape[0]) {
            throw std::invalid_argument(std::string(name) + " size mismatch.");
        }
        return;
    }
    if (shape.size() == 2 && (shape[0] == 1 || shape[1] == 1)) {
        if (tensor.size() != shape[0] * shape[1]) {
            throw std::invalid_argument(std::string(name) + " size mismatch.");
        }
        return;
    }
    throw std::invalid_argument(std::string(name) + " rank mismatch.");
}

void validate_linear_weight(const Tensor& weight, const char* name) {
    validate_rank(weight, 2, name);
    int expected = weight.shape()[0] * weight.shape()[1];
    if (weight.size() != expected) {
        throw std::invalid_argument(std::string(name) + " size mismatch.");
    }
}

void validate_bias(const Tensor& bias, const char* name) {
    validate_rank(bias, 1, name);
    if (bias.size() != bias.shape()[0]) {
        throw std::invalid_argument(std::string(name) + " size mismatch.");
    }
}

template <typename Traits>
void launch_linear(cublasHandle_t handle,
                   const void* weight_ptr,
                   const void* input_ptr,
                   void* output_ptr,
                   int rows,
                   int in_dim,
                   int out_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmEx(handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     out_dim, rows, in_dim,
                     &alpha,
                     weight_ptr, Traits::kCudaType, in_dim,
                     input_ptr, Traits::kCudaType, in_dim,
                     &beta,
                     output_ptr, Traits::kCudaType, out_dim,
                     Traits::kComputeType,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx linear");
}

template <typename Traits>
void maybe_add_bias(cudaStream_t stream,
                    void* output_ptr,
                    const void* bias_ptr,
                    int rows,
                    int cols) {
    if (bias_ptr == nullptr || cols <= 0) {
        return;
    }
    int total = rows * cols;
    constexpr int kThreads = 256;
    int blocks = (total + kThreads - 1) / kThreads;
    add_bias_kernel<typename Traits::DeviceType><<<blocks, kThreads, 0, stream>>>(
        static_cast<typename Traits::DeviceType*>(output_ptr),
        static_cast<const typename Traits::DeviceType*>(bias_ptr),
        rows, cols);
    cuda_check(cudaGetLastError(), "add_bias_kernel");
}

size_t next_capacity_bytes(size_t current_capacity, size_t required_bytes) {
    if (required_bytes == 0) {
        return 0;
    }
    size_t capacity = current_capacity;
    if (capacity == 0) {
        capacity = std::max(required_bytes, static_cast<size_t>(4096));
    }
    while (capacity < required_bytes) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) {
            capacity = required_bytes;
            break;
        }
        capacity *= 2;
    }
    return capacity;
}

void ensure_device_buffer(DeviceBuffer& buffer,
                          size_t required_bytes,
                          std::uint64_t* realloc_counter = nullptr) {
    if (required_bytes == 0 || buffer.bytes() >= required_bytes) {
        return;
    }
    const size_t new_capacity = next_capacity_bytes(buffer.bytes(), required_bytes);
    buffer.reset(new_capacity);
    if (realloc_counter != nullptr) {
        *realloc_counter += 1;
    }
}

template <typename Traits>
const void* get_or_upload_tensor(const Tensor& tensor,
                                 TensorUploadCache& cache,
                                 cudaStream_t stream,
                                 const char* error_msg) {
    const size_t bytes = static_cast<size_t>(tensor.size()) * Traits::kElementBytes;
    if (bytes == 0) {
        return nullptr;
    }
    const void* host_ptr = tensor.data().data();
    bool needs_upload = false;
    if (cache.device.bytes() < bytes) {
        cache.device.reset(next_capacity_bytes(cache.device.bytes(), bytes));
        needs_upload = true;
    }
    if (cache.host_ptr != host_ptr || cache.bytes != bytes) {
        needs_upload = true;
    }
    if (needs_upload) {
        cuda_check(cudaMemcpyAsync(cache.device.data(), host_ptr, bytes,
                                   cudaMemcpyHostToDevice, stream),
                   error_msg);
        cache.host_ptr = host_ptr;
        cache.bytes = bytes;
    }
    return cache.device.data();
}

const int* get_or_upload_int_vector(const std::vector<int>& input,
                                    IntVectorUploadCache& cache,
                                    cudaStream_t stream,
                                    const char* error_msg,
                                    std::uint64_t* upload_counter = nullptr) {
    if (input.empty()) {
        return nullptr;
    }
    const size_t bytes = static_cast<size_t>(input.size()) * sizeof(int);
    bool needs_upload = false;
    std::uint64_t checksum = 1469598103934665603ull;
    for (int value : input) {
        checksum ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
        checksum *= 1099511628211ull;
    }
    if (cache.device.bytes() < bytes) {
        cache.device.reset(next_capacity_bytes(cache.device.bytes(), bytes));
        needs_upload = true;
    }
    if (cache.host_ptr != input.data() || cache.count != input.size() || cache.checksum != checksum) {
        needs_upload = true;
    }
    if (needs_upload) {
        cuda_check(cudaMemcpyAsync(cache.device.data(), input.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   error_msg);
        cache.host_ptr = input.data();
        cache.count = input.size();
        cache.checksum = checksum;
        if (upload_counter != nullptr) {
            *upload_counter += 1;
        }
    }
    return static_cast<const int*>(cache.device.data());
}

int next_capacity_len(int current_capacity, int required_len) {
    if (required_len <= 0) {
        return 0;
    }
    int capacity = std::max(current_capacity, 0);
    if (capacity == 0) {
        capacity = std::max(required_len, 8);
    }
    while (capacity < required_len) {
        if (capacity > std::numeric_limits<int>::max() / 2) {
            capacity = required_len;
            break;
        }
        capacity *= 2;
    }
    return capacity;
}

void ensure_sample_capacity(SelfAttnCudaState::Impl& impl, int min_size) {
    if (min_size <= 0) {
        return;
    }
    if (static_cast<int>(impl.samples.size()) < min_size) {
        impl.samples.resize(min_size);
    }
}

template <typename Traits>
void append_sample_cache(SampleCache& sample,
                         const void* k_slice,
                         const void* v_slice,
                         int num_heads_kv,
                         int seq_len,
                         int head_dim,
                         cudaStream_t stream) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const size_t elem_bytes = Traits::kElementBytes;
    const int old_len = sample.len;
    const int new_len = old_len + seq_len;
    const size_t row_elems = static_cast<size_t>(num_heads_kv) * head_dim;
    if (new_len > sample.capacity_len) {
        const int new_capacity = next_capacity_len(sample.capacity_len, new_len);
        const size_t total_new_bytes = static_cast<size_t>(new_capacity) * row_elems * elem_bytes;

        DeviceBuffer new_k(total_new_bytes);
        DeviceBuffer new_v(total_new_bytes);
        if (old_len > 0) {
            const size_t old_copy_bytes = static_cast<size_t>(old_len) * row_elems * elem_bytes;
            cuda_check(cudaMemcpyAsync(new_k.data(), sample.k.data(), old_copy_bytes, cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync cache k old");
            cuda_check(cudaMemcpyAsync(new_v.data(), sample.v.data(), old_copy_bytes, cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync cache v old");
        }
        sample.k = std::move(new_k);
        sample.v = std::move(new_v);
        sample.capacity_len = new_capacity;
    }

    const int append_total = seq_len * num_heads_kv * head_dim;
    const int append_blocks = (append_total + kThreads - 1) / kThreads;
    append_kv_interleaved_kernel<DeviceType><<<append_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(k_slice),
        static_cast<DeviceType*>(sample.k.data()),
        old_len, seq_len, num_heads_kv, head_dim);
    append_kv_interleaved_kernel<DeviceType><<<append_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(v_slice),
        static_cast<DeviceType*>(sample.v.data()),
        old_len, seq_len, num_heads_kv, head_dim);
    cuda_check(cudaGetLastError(), "append_kv_interleaved_kernel");
    sample.len = new_len;
}

struct ActiveCacheInfo {
    int max_len{0};
    std::vector<int> active_lens;
};

template <typename Traits>
ActiveCacheInfo append_kv_cache(SelfAttnCudaState::Impl& impl,
                                const std::vector<int>& sample_ids,
                                int batch,
                                int seq_len,
                                int num_heads_kv,
                                int head_dim,
                                const DeviceBuffer& d_k,
                                const DeviceBuffer& d_v,
                                cudaStream_t stream) {
    const size_t elem_bytes = Traits::kElementBytes;
    const size_t sample_bytes = static_cast<size_t>(num_heads_kv) * seq_len * head_dim * elem_bytes;
    for (int i = 0; i < batch; ++i) {
        int sample_id = sample_ids[i];
        if (sample_id < 0) {
            throw std::invalid_argument("sample_id must be non-negative.");
        }
        ensure_sample_capacity(impl, sample_id + 1);
        SampleCache& sample = impl.samples[sample_id];
        const char* k_slice = static_cast<const char*>(d_k.data()) + static_cast<size_t>(i) * sample_bytes;
        const char* v_slice = static_cast<const char*>(d_v.data()) + static_cast<size_t>(i) * sample_bytes;
        append_sample_cache<Traits>(sample, k_slice, v_slice, num_heads_kv, seq_len, head_dim, stream);
    }

    ActiveCacheInfo active_cache{};
    active_cache.active_lens.reserve(sample_ids.size());
    for (int sample_id : sample_ids) {
        if (sample_id < 0 || sample_id >= static_cast<int>(impl.samples.size())) {
            throw std::out_of_range("sample_id out of cache range.");
        }
        const int len = impl.samples[sample_id].len;
        active_cache.active_lens.push_back(len);
        active_cache.max_len = std::max(active_cache.max_len, len);
    }
    return active_cache;
}

template <typename Traits>
void build_cache_batch(const SelfAttnCudaState::Impl& impl,
                       const std::vector<int>& sample_ids,
                       int num_heads_kv,
                       int total_len,
                       int head_dim,
                       DeviceBuffer& d_cache_k_batch,
                       DeviceBuffer& d_cache_v_batch,
                       cudaStream_t stream) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const size_t elem_bytes = Traits::kElementBytes;
    const int batch = static_cast<int>(sample_ids.size());
    const size_t sample_bytes = static_cast<size_t>(num_heads_kv) * total_len * head_dim * elem_bytes;
    ensure_device_buffer(d_cache_k_batch, sample_bytes * batch);
    ensure_device_buffer(d_cache_v_batch, sample_bytes * batch);
    for (int i = 0; i < batch; ++i) {
        int sample_id = sample_ids[i];
        const SampleCache& sample = impl.samples[sample_id];
        if (sample.len > total_len) {
            throw std::invalid_argument("Cache length exceeds batched cache capacity.");
        }
        char* dst_k = static_cast<char*>(d_cache_k_batch.data()) + static_cast<size_t>(i) * sample_bytes;
        char* dst_v = static_cast<char*>(d_cache_v_batch.data()) + static_cast<size_t>(i) * sample_bytes;
        cuda_check(cudaMemsetAsync(dst_k, 0, sample_bytes, stream), "cudaMemsetAsync cache k batch");
        cuda_check(cudaMemsetAsync(dst_v, 0, sample_bytes, stream), "cudaMemsetAsync cache v batch");
        if (sample.len <= 0) {
            continue;
        }
        const int pack_total = sample.len * num_heads_kv * head_dim;
        const int pack_blocks = (pack_total + kThreads - 1) / kThreads;
        pack_cache_interleaved_to_heads_kernel<DeviceType><<<pack_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(sample.k.data()),
            reinterpret_cast<DeviceType*>(dst_k),
            sample.len, total_len, num_heads_kv, head_dim);
        pack_cache_interleaved_to_heads_kernel<DeviceType><<<pack_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(sample.v.data()),
            reinterpret_cast<DeviceType*>(dst_v),
            sample.len, total_len, num_heads_kv, head_dim);
    }
    cuda_check(cudaGetLastError(), "pack_cache_interleaved_to_heads_kernel");
}

template <typename Traits>
void build_cache_batch_interleaved(const SelfAttnCudaState::Impl& impl,
                                   const std::vector<int>& sample_ids,
                                   int num_heads_kv,
                                   int total_len,
                                   int head_dim,
                                   DeviceBuffer& d_cache_k_batch,
                                   DeviceBuffer& d_cache_v_batch,
                                   cudaStream_t stream) {
    const size_t elem_bytes = Traits::kElementBytes;
    const int batch = static_cast<int>(sample_ids.size());
    const size_t sample_bytes = static_cast<size_t>(num_heads_kv) * total_len * head_dim * elem_bytes;
    ensure_device_buffer(d_cache_k_batch, sample_bytes * batch);
    ensure_device_buffer(d_cache_v_batch, sample_bytes * batch);
    for (int i = 0; i < batch; ++i) {
        int sample_id = sample_ids[i];
        const SampleCache& sample = impl.samples[sample_id];
        if (sample.len > total_len) {
            throw std::invalid_argument("Cache length exceeds interleaved batch capacity.");
        }
        char* dst_k = static_cast<char*>(d_cache_k_batch.data()) + static_cast<size_t>(i) * sample_bytes;
        char* dst_v = static_cast<char*>(d_cache_v_batch.data()) + static_cast<size_t>(i) * sample_bytes;
        cuda_check(cudaMemsetAsync(dst_k, 0, sample_bytes, stream), "cudaMemsetAsync cache k batch interleaved");
        cuda_check(cudaMemsetAsync(dst_v, 0, sample_bytes, stream), "cudaMemsetAsync cache v batch interleaved");
        if (sample.len <= 0) {
            continue;
        }
        const size_t copy_bytes = static_cast<size_t>(num_heads_kv) * sample.len * head_dim * elem_bytes;
        cuda_check(cudaMemcpyAsync(dst_k, sample.k.data(), copy_bytes, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync cache k batch interleaved");
        cuda_check(cudaMemcpyAsync(dst_v, sample.v.data(), copy_bytes, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync cache v batch interleaved");
    }
}

template <typename Traits>
void launch_qk_grouped_batched_gemm(cublasHandle_t handle,
                                    const void* d_cache_k,
                                    const void* d_q,
                                    void* d_scores,
                                    int batch,
                                    int num_heads_kv,
                                    int repeat_factor,
                                    int seq_len,
                                    int total_len,
                                    int head_dim,
                                    int cache_leading_dim,
                                    long long cache_instance_stride_elems) {
    const int grouped_seq = seq_len * repeat_factor;
    const int batch_count = batch * num_heads_kv;
    const long long stride_a = cache_instance_stride_elems;
    const long long stride_b = static_cast<long long>(grouped_seq) * head_dim;
    const long long stride_c = static_cast<long long>(grouped_seq) * total_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   total_len, grouped_seq, head_dim,
                                   &alpha,
                                   d_cache_k, Traits::kCudaType, cache_leading_dim, stride_a,
                                   d_q, Traits::kCudaType, head_dim, stride_b,
                                   &beta,
                                   d_scores, Traits::kCudaType, total_len, stride_c,
                                   batch_count, Traits::kComputeType,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx qk grouped");
}

template <typename Traits>
void launch_av_grouped_batched_gemm(cublasHandle_t handle,
                                    const void* d_scores,
                                    const void* d_cache_v,
                                    void* d_context,
                                    int batch,
                                    int num_heads_kv,
                                    int repeat_factor,
                                    int seq_len,
                                    int total_len,
                                    int head_dim,
                                    int cache_leading_dim,
                                    long long cache_instance_stride_elems) {
    const int grouped_seq = seq_len * repeat_factor;
    const int batch_count = batch * num_heads_kv;
    const long long stride_a = cache_instance_stride_elems;
    const long long stride_b = static_cast<long long>(grouped_seq) * total_len;
    const long long stride_c = static_cast<long long>(grouped_seq) * head_dim;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   head_dim, grouped_seq, total_len,
                                   &alpha,
                                   d_cache_v, Traits::kCudaType, cache_leading_dim, stride_a,
                                   d_scores, Traits::kCudaType, total_len, stride_b,
                                   &beta,
                                   d_context, Traits::kCudaType, head_dim, stride_c,
                                   batch_count, Traits::kComputeType,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx av grouped");
}

bool can_reuse_decode_graph(const DecodeGraphCache& graph,
                            int batch,
                            int sample_id,
                            int num_heads,
                            int head_dim,
                            int hidden_dim,
                            int q_out_dim,
                            int k_out_dim,
                            int v_out_dim,
                            int o_out_dim,
                            int repeat_factor,
                            int score_capacity,
                            int pad_size,
                            bool use_qk_norm,
                            const void* cache_k_ptr,
                            const void* cache_v_ptr,
                            const int* d_pad_ptr,
                            const void* d_norm_weight,
                            const void* d_q_norm_weight,
                            const void* d_k_norm_weight,
                            const void* q_weight_ptr,
                            const void* k_weight_ptr,
                            const void* v_weight_ptr,
                            const void* o_weight_ptr,
                            const void* d_q_bias,
                            const void* d_k_bias,
                            const void* d_v_bias,
                            const void* d_o_bias) {
    return graph.exec != nullptr &&
           graph.batch == batch &&
           graph.sample_id == sample_id &&
           graph.num_heads == num_heads &&
           graph.head_dim == head_dim &&
           graph.hidden_dim == hidden_dim &&
           graph.q_out_dim == q_out_dim &&
           graph.k_out_dim == k_out_dim &&
           graph.v_out_dim == v_out_dim &&
           graph.o_out_dim == o_out_dim &&
           graph.repeat_factor == repeat_factor &&
           graph.score_capacity == score_capacity &&
           graph.pad_size == pad_size &&
           graph.use_qk_norm == use_qk_norm &&
           graph.cache_k_ptr == cache_k_ptr &&
           graph.cache_v_ptr == cache_v_ptr &&
           graph.d_pad_ptr == d_pad_ptr &&
           graph.d_norm_weight_ptr == d_norm_weight &&
           graph.d_q_norm_weight_ptr == d_q_norm_weight &&
           graph.d_k_norm_weight_ptr == d_k_norm_weight &&
           graph.q_weight_ptr == q_weight_ptr &&
           graph.k_weight_ptr == k_weight_ptr &&
           graph.v_weight_ptr == v_weight_ptr &&
           graph.o_weight_ptr == o_weight_ptr &&
           graph.d_q_bias_ptr == d_q_bias &&
           graph.d_k_bias_ptr == d_k_bias &&
           graph.d_v_bias_ptr == d_v_bias &&
           graph.d_o_bias_ptr == d_o_bias;
}

void fill_decode_graph_signature(DecodeGraphCache& graph,
                                 int batch,
                                 int sample_id,
                                 int num_heads,
                                 int head_dim,
                                 int hidden_dim,
                                 int q_out_dim,
                                 int k_out_dim,
                                 int v_out_dim,
                                 int o_out_dim,
                                 int repeat_factor,
                                 int score_capacity,
                                 int pad_size,
                                 bool use_qk_norm,
                                 const void* cache_k_ptr,
                                 const void* cache_v_ptr,
                                 const int* d_pad_ptr,
                                 const void* d_norm_weight,
                                 const void* d_q_norm_weight,
                                 const void* d_k_norm_weight,
                                 const void* q_weight_ptr,
                                 const void* k_weight_ptr,
                                 const void* v_weight_ptr,
                                 const void* o_weight_ptr,
                                 const void* d_q_bias,
                                 const void* d_k_bias,
                                 const void* d_v_bias,
                                 const void* d_o_bias) {
    graph.batch = batch;
    graph.sample_id = sample_id;
    graph.num_heads = num_heads;
    graph.head_dim = head_dim;
    graph.hidden_dim = hidden_dim;
    graph.q_out_dim = q_out_dim;
    graph.k_out_dim = k_out_dim;
    graph.v_out_dim = v_out_dim;
    graph.o_out_dim = o_out_dim;
    graph.repeat_factor = repeat_factor;
    graph.score_capacity = score_capacity;
    graph.pad_size = pad_size;
    graph.use_qk_norm = use_qk_norm;
    graph.cache_k_ptr = cache_k_ptr;
    graph.cache_v_ptr = cache_v_ptr;
    graph.d_pad_ptr = d_pad_ptr;
    graph.d_norm_weight_ptr = d_norm_weight;
    graph.d_q_norm_weight_ptr = d_q_norm_weight;
    graph.d_k_norm_weight_ptr = d_k_norm_weight;
    graph.q_weight_ptr = q_weight_ptr;
    graph.k_weight_ptr = k_weight_ptr;
    graph.v_weight_ptr = v_weight_ptr;
    graph.o_weight_ptr = o_weight_ptr;
    graph.d_q_bias_ptr = d_q_bias;
    graph.d_k_bias_ptr = d_k_bias;
    graph.d_v_bias_ptr = d_v_bias;
    graph.d_o_bias_ptr = d_o_bias;
}

} // namespace

} // namespace ops
} // namespace cuda
} // namespace easy_llm

#endif // EASY_LLM_CUDA_OPS_SELF_ATTN_DETAIL_CUH
