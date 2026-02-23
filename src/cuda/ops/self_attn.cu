#include "cuda/ops/self_attn.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

#include "self_attn_detail.cuh"

namespace easy_llm {
namespace cuda {
namespace ops {

namespace {

struct CacheBatchView {
    const void* cache_k_ptr{nullptr};
    const void* cache_v_ptr{nullptr};
    int score_capacity{0};
    int cache_leading_dim{0};
    long long cache_instance_stride_elems{0};
};

int pick_fused_threads(int head_dim, int max_threads) {
    int fused_threads = 32;
    while (fused_threads < head_dim && fused_threads < max_threads) {
        fused_threads <<= 1;
    }
    return fused_threads;
}

size_t fused_attention_shared_bytes(int fused_threads, int head_dim) {
    return static_cast<size_t>(fused_threads + 2 * head_dim + 2) * sizeof(float);
}

template <typename Traits>
CacheBatchView prepare_cache_batch_view(SelfAttnCudaState::Impl& impl,
                                        ForwardScratchBuffers& scratch,
                                        const std::vector<int>& sample_ids,
                                        int batch,
                                        int num_heads_kv,
                                        int total_len,
                                        int head_dim,
                                        bool use_interleaved_batch,
                                        bool prefer_single_capacity_len,
                                        cudaStream_t stream) {
    CacheBatchView view{};
    view.score_capacity = total_len;
    view.cache_leading_dim = head_dim;
    view.cache_instance_stride_elems = static_cast<long long>(total_len) * head_dim;

    if (batch == 1) {
        const SampleCache& single = impl.samples[sample_ids[0]];
        if (single.len != total_len) {
            throw std::invalid_argument("self_attn_forward_cuda: cache length mismatch for batch=1.");
        }
        view.cache_k_ptr = single.k.data();
        view.cache_v_ptr = single.v.data();
        if (prefer_single_capacity_len) {
            view.score_capacity = std::max(single.capacity_len, total_len);
        }
        view.cache_leading_dim = num_heads_kv * head_dim;
        view.cache_instance_stride_elems = head_dim;
        return view;
    }

    if (use_interleaved_batch) {
        build_cache_batch_interleaved<Traits>(impl, sample_ids, num_heads_kv, total_len, head_dim,
                                              scratch.cache_k_batch, scratch.cache_v_batch, stream);
    } else {
        build_cache_batch<Traits>(impl, sample_ids, num_heads_kv, total_len, head_dim,
                                  scratch.cache_k_batch, scratch.cache_v_batch, stream);
    }
    view.cache_k_ptr = scratch.cache_k_batch.data();
    view.cache_v_ptr = scratch.cache_v_batch.data();
    return view;
}

void prepare_attention_mask_inputs(SelfAttnCudaState::Impl& impl,
                                   ForwardScratchBuffers& scratch,
                                   const std::vector<int>& sample_ids,
                                   const std::vector<int>& pad_lens_by_sample,
                                   cudaStream_t stream,
                                   std::uint64_t* scratch_realloc_counter,
                                   const int*& d_sample_ids,
                                   const int*& d_pad_ptr,
                                   int& pad_size) {
    d_sample_ids = nullptr;
    d_pad_ptr = nullptr;
    pad_size = static_cast<int>(pad_lens_by_sample.size());
    if (pad_size <= 0) {
        return;
    }
    ensure_device_buffer(scratch.sample_ids, static_cast<size_t>(sample_ids.size()) * sizeof(int), scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(scratch.sample_ids.data(), sample_ids.data(), static_cast<size_t>(sample_ids.size()) * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync sample_ids");
    d_sample_ids = static_cast<const int*>(scratch.sample_ids.data());
    d_pad_ptr = get_or_upload_int_vector(
        pad_lens_by_sample,
        impl.pad_lens_cache,
        stream,
        "cudaMemcpyAsync pad_lens",
        &impl.stats.pad_lens_uploads);
}

template <typename Traits>
void launch_output_projection(cublasHandle_t handle,
                              cudaStream_t stream,
                              ForwardScratchBuffers& scratch,
                              int batch,
                              int seq_len,
                              int num_heads,
                              int head_dim,
                              int rows,
                              int hidden_dim,
                              int o_out_dim,
                              const void* o_weight_ptr,
                              const void* d_o_bias,
                              std::uint64_t* scratch_realloc_counter,
                              const char* merge_error) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;

    ensure_device_buffer(scratch.merged, static_cast<size_t>(rows) * hidden_dim * Traits::kElementBytes, scratch_realloc_counter);
    int merge_total = rows * hidden_dim;
    int merge_blocks = (merge_total + kThreads - 1) / kThreads;
    merge_heads_kernel<DeviceType><<<merge_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.context.data()),
        static_cast<DeviceType*>(scratch.merged.data()),
        batch, seq_len, num_heads, head_dim);
    cuda_check(cudaGetLastError(), merge_error);

    ensure_device_buffer(scratch.output, static_cast<size_t>(rows) * o_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    launch_linear<Traits>(handle, o_weight_ptr, scratch.merged.data(), scratch.output.data(), rows, hidden_dim, o_out_dim);
    maybe_add_bias<Traits>(stream, scratch.output.data(), d_o_bias, rows, o_out_dim);
}

template <typename Traits>
Tensor copy_output_to_host(cudaStream_t stream,
                           ForwardScratchBuffers& scratch,
                           int rows,
                           int o_out_dim,
                           int batch,
                           int seq_len,
                           const char* copy_error,
                           const char* sync_error) {
    Tensor output(rows * o_out_dim);
    cuda_check(cudaMemcpyAsync(output.data().data(), scratch.output.data(),
                               static_cast<size_t>(rows) * o_out_dim * Traits::kElementBytes,
                               cudaMemcpyDeviceToHost, stream),
               copy_error);
    cuda_check(cudaStreamSynchronize(stream), sync_error);
    output.reshape({batch, seq_len, o_out_dim});
    return output;
}

struct ForwardShapeInfo {
    int batch{0};
    int seq_len{0};
    int hidden_dim{0};
    int rows{0};
    int repeat_factor{0};
    int q_out_dim{0};
    int k_out_dim{0};
    int v_out_dim{0};
    int o_out_dim{0};
};

struct ForwardDeviceResources {
    const void* d_norm_weight{nullptr};
    const void* d_q_norm_weight{nullptr};
    const void* d_k_norm_weight{nullptr};
    const void* q_weight_ptr{nullptr};
    const void* k_weight_ptr{nullptr};
    const void* v_weight_ptr{nullptr};
    const void* o_weight_ptr{nullptr};
    const void* d_q_bias{nullptr};
    const void* d_k_bias{nullptr};
    const void* d_v_bias{nullptr};
    const void* d_o_bias{nullptr};
};

struct AttentionMaskInputs {
    const int* d_sample_ids{nullptr};
    const int* d_pad_ptr{nullptr};
    int pad_size{0};
};

struct DecodeAttentionResources {
    const void* cache_k_ptr{nullptr};
    const void* cache_v_ptr{nullptr};
    int score_capacity{0};
    int cache_leading_dim{0};
    long long cache_instance_stride_elems{0};
    const int* d_active_len{nullptr};
    AttentionMaskInputs mask_inputs{};
};

ForwardShapeInfo validate_forward_shape_and_state(const Tensor& input,
                                                  const std::vector<int>& sample_ids,
                                                  const std::vector<int>& offsets,
                                                  const SelfAttnCudaParams& params,
                                                  const Tensor& norm_weight,
                                                  const Tensor& q_norm_weight,
                                                  const Tensor& k_norm_weight,
                                                  const Tensor& q_weight,
                                                  const Tensor& q_bias,
                                                  const Tensor& k_weight,
                                                  const Tensor& k_bias,
                                                  const Tensor& v_weight,
                                                  const Tensor& v_bias,
                                                  const Tensor& o_weight,
                                                  const Tensor& o_bias,
                                                  SelfAttnCudaState::Impl& impl) {
    if (params.hidden_dim <= 0 || params.num_heads <= 0 || params.num_heads_kv <= 0 ||
        params.head_dim <= 0 || params.rope_theta <= 0.0f) {
        throw std::invalid_argument("self_attn_forward_cuda: invalid params.");
    }
    if (params.num_heads % params.num_heads_kv != 0) {
        throw std::invalid_argument("self_attn_forward_cuda: num_heads must be divisible by num_heads_kv.");
    }
    if (input.shape().size() != 3) {
        throw std::invalid_argument("self_attn_forward_cuda: input must be [batch, seq, hidden].");
    }

    ForwardShapeInfo shape{};
    shape.batch = input.shape()[0];
    shape.seq_len = input.shape()[1];
    shape.hidden_dim = input.shape()[2];
    shape.rows = shape.batch * shape.seq_len;
    shape.repeat_factor = params.num_heads / params.num_heads_kv;

    if (shape.hidden_dim != params.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: input hidden_dim mismatch.");
    }
    if (input.size() != shape.batch * shape.seq_len * shape.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: input size mismatch.");
    }
    if (sample_ids.size() != static_cast<size_t>(shape.batch)) {
        throw std::invalid_argument("self_attn_forward_cuda: sample_ids size must match batch.");
    }
    if (offsets.size() != static_cast<size_t>(shape.batch)) {
        throw std::invalid_argument("self_attn_forward_cuda: offsets size must match batch.");
    }

    validate_vector_like(norm_weight, "norm_weight");
    if (norm_weight.size() != shape.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: norm_weight size mismatch.");
    }
    if (params.use_qk_norm) {
        validate_vector_like(q_norm_weight, "q_norm_weight");
        validate_vector_like(k_norm_weight, "k_norm_weight");
        if (q_norm_weight.size() != params.head_dim || k_norm_weight.size() != params.head_dim) {
            throw std::invalid_argument("self_attn_forward_cuda: q/k norm weight size mismatch.");
        }
    }
    validate_linear_weight(q_weight, "q_weight");
    validate_linear_weight(k_weight, "k_weight");
    validate_linear_weight(v_weight, "v_weight");
    validate_linear_weight(o_weight, "o_weight");
    validate_bias(q_bias, "q_bias");
    validate_bias(k_bias, "k_bias");
    validate_bias(v_bias, "v_bias");
    validate_bias(o_bias, "o_bias");

    shape.q_out_dim = q_weight.shape()[0];
    shape.k_out_dim = k_weight.shape()[0];
    shape.v_out_dim = v_weight.shape()[0];
    shape.o_out_dim = o_weight.shape()[0];

    if (q_weight.shape()[1] != shape.hidden_dim || k_weight.shape()[1] != shape.hidden_dim || v_weight.shape()[1] != shape.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: q/k/v weight input dim mismatch.");
    }
    const int q_hidden_dim = params.num_heads * params.head_dim;
    if (shape.q_out_dim != q_hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: q_weight output dim mismatch.");
    }
    if (shape.k_out_dim != params.num_heads_kv * params.head_dim ||
        shape.v_out_dim != params.num_heads_kv * params.head_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: k/v weight output dim mismatch.");
    }
    if (o_weight.shape()[1] != shape.q_out_dim || shape.o_out_dim != shape.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: o_weight shape mismatch.");
    }
    if (params.head_dim % 2 != 0) {
        throw std::invalid_argument("self_attn_forward_cuda: head_dim must be even for RoPE.");
    }

    if (impl.num_heads == 0) {
        impl.num_heads = params.num_heads;
        impl.head_dim = params.head_dim;
    } else if (impl.num_heads != params.num_heads || impl.head_dim != params.head_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: state head config mismatch.");
    }

    int max_sample_id = -1;
    for (int sample_id : sample_ids) {
        if (sample_id < 0) {
            throw std::invalid_argument("self_attn_forward_cuda: sample_id must be non-negative.");
        }
        max_sample_id = std::max(max_sample_id, sample_id);
    }
    ensure_sample_capacity(impl, max_sample_id + 1);
    return shape;
}

template <typename Traits>
ForwardDeviceResources prepare_forward_device_resources(CudaContext& ctx,
                                                        SelfAttnCudaState::Impl& impl,
                                                        ForwardScratchBuffers& scratch,
                                                        const SelfAttnCudaParams& params,
                                                        const ForwardShapeInfo& shape,
                                                        const Tensor& input,
                                                        const Tensor& norm_weight,
                                                        const Tensor& q_norm_weight,
                                                        const Tensor& k_norm_weight,
                                                        const Tensor& q_weight,
                                                        const Tensor& q_bias,
                                                        const Tensor& k_weight,
                                                        const Tensor& k_bias,
                                                        const Tensor& v_weight,
                                                        const Tensor& v_bias,
                                                        const Tensor& o_weight,
                                                        const Tensor& o_bias,
                                                        cudaStream_t stream,
                                                        cublasHandle_t handle,
                                                        std::uint64_t* scratch_realloc_counter) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;

    ForwardDeviceResources resources{};
    const size_t input_bytes = static_cast<size_t>(shape.rows) * shape.hidden_dim * Traits::kElementBytes;

    ensure_device_buffer(scratch.input, input_bytes, scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(scratch.input.data(), input.data().data(), input_bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync input");

    resources.d_norm_weight = get_or_upload_tensor<Traits>(
        norm_weight, impl.norm_weight_cache, stream, "cudaMemcpyAsync norm_weight");
    if (params.use_qk_norm) {
        resources.d_q_norm_weight = get_or_upload_tensor<Traits>(
            q_norm_weight, impl.q_norm_weight_cache, stream, "cudaMemcpyAsync q_norm_weight");
        resources.d_k_norm_weight = get_or_upload_tensor<Traits>(
            k_norm_weight, impl.k_norm_weight_cache, stream, "cudaMemcpyAsync k_norm_weight");
    }

    ensure_device_buffer(scratch.norm_out, input_bytes, scratch_realloc_counter);
    rms_norm_kernel<DeviceType><<<shape.rows, kThreads, kThreads * sizeof(float), stream>>>(
        static_cast<const DeviceType*>(scratch.input.data()),
        static_cast<const DeviceType*>(resources.d_norm_weight),
        static_cast<DeviceType*>(scratch.norm_out.data()),
        shape.rows, shape.hidden_dim, 1e-6f);
    cuda_check(cudaGetLastError(), "rms_norm_kernel");

    WeightEntry& q_weight_entry = ctx.cache().get_or_upload(q_weight, stream);
    WeightEntry& k_weight_entry = ctx.cache().get_or_upload(k_weight, stream);
    WeightEntry& v_weight_entry = ctx.cache().get_or_upload(v_weight, stream);
    WeightEntry& o_weight_entry = ctx.cache().get_or_upload(o_weight, stream);
    resources.q_weight_ptr = q_weight_entry.device_ptr;
    resources.k_weight_ptr = k_weight_entry.device_ptr;
    resources.v_weight_ptr = v_weight_entry.device_ptr;
    resources.o_weight_ptr = o_weight_entry.device_ptr;

    ensure_device_buffer(scratch.q_proj, static_cast<size_t>(shape.rows) * shape.q_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.k_proj, static_cast<size_t>(shape.rows) * shape.k_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.v_proj, static_cast<size_t>(shape.rows) * shape.v_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    launch_linear<Traits>(handle, resources.q_weight_ptr, scratch.norm_out.data(), scratch.q_proj.data(), shape.rows, shape.hidden_dim, shape.q_out_dim);
    launch_linear<Traits>(handle, resources.k_weight_ptr, scratch.norm_out.data(), scratch.k_proj.data(), shape.rows, shape.hidden_dim, shape.k_out_dim);
    launch_linear<Traits>(handle, resources.v_weight_ptr, scratch.norm_out.data(), scratch.v_proj.data(), shape.rows, shape.hidden_dim, shape.v_out_dim);

    if (q_bias.size() == shape.q_out_dim) {
        resources.d_q_bias = get_or_upload_tensor<Traits>(q_bias, impl.q_bias_cache, stream, "cudaMemcpyAsync q_bias");
    }
    if (k_bias.size() == shape.k_out_dim) {
        resources.d_k_bias = get_or_upload_tensor<Traits>(k_bias, impl.k_bias_cache, stream, "cudaMemcpyAsync k_bias");
    }
    if (v_bias.size() == shape.v_out_dim) {
        resources.d_v_bias = get_or_upload_tensor<Traits>(v_bias, impl.v_bias_cache, stream, "cudaMemcpyAsync v_bias");
    }
    if (o_bias.size() == shape.o_out_dim) {
        resources.d_o_bias = get_or_upload_tensor<Traits>(o_bias, impl.o_bias_cache, stream, "cudaMemcpyAsync o_bias");
    }

    return resources;
}

template <typename Traits>
void apply_qkv_projection_biases(cudaStream_t stream,
                                 ForwardScratchBuffers& scratch,
                                 const ForwardShapeInfo& shape,
                                 const ForwardDeviceResources& resources) {
    maybe_add_bias<Traits>(stream, scratch.q_proj.data(), resources.d_q_bias, shape.rows, shape.q_out_dim);
    maybe_add_bias<Traits>(stream, scratch.k_proj.data(), resources.d_k_bias, shape.rows, shape.k_out_dim);
    maybe_add_bias<Traits>(stream, scratch.v_proj.data(), resources.d_v_bias, shape.rows, shape.v_out_dim);
}

template <typename Traits>
void apply_qk_norm_if_enabled(cudaStream_t stream,
                              ForwardScratchBuffers& scratch,
                              const SelfAttnCudaParams& params,
                              const ForwardShapeInfo& shape,
                              const ForwardDeviceResources& resources) {
    if (!params.use_qk_norm) {
        return;
    }
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const int q_vectors = shape.batch * params.num_heads * shape.seq_len;
    const int k_vectors = shape.batch * params.num_heads_kv * shape.seq_len;
    rms_norm_kernel<DeviceType><<<q_vectors, kThreads, kThreads * sizeof(float), stream>>>(
        static_cast<const DeviceType*>(scratch.q.data()),
        static_cast<const DeviceType*>(resources.d_q_norm_weight),
        static_cast<DeviceType*>(scratch.q.data()),
        q_vectors, params.head_dim, 1e-6f);
    rms_norm_kernel<DeviceType><<<k_vectors, kThreads, kThreads * sizeof(float), stream>>>(
        static_cast<const DeviceType*>(scratch.k.data()),
        static_cast<const DeviceType*>(resources.d_k_norm_weight),
        static_cast<DeviceType*>(scratch.k.data()),
        k_vectors, params.head_dim, 1e-6f);
    cuda_check(cudaGetLastError(), "rms_norm_kernel qk");
}

void upload_offsets_to_device(ForwardScratchBuffers& scratch,
                              const std::vector<int>& offsets,
                              int batch,
                              cudaStream_t stream,
                              std::uint64_t* scratch_realloc_counter) {
    ensure_device_buffer(scratch.offsets, static_cast<size_t>(batch) * sizeof(int), scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(scratch.offsets.data(), offsets.data(), static_cast<size_t>(batch) * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync offsets");
}

void ensure_rope_inv_freq_device(SelfAttnCudaState::Impl& impl,
                                 int head_dim,
                                 float rope_theta,
                                 cudaStream_t stream,
                                 std::uint64_t* scratch_realloc_counter) {
    if (head_dim <= 0 || (head_dim % 2) != 0) {
        throw std::invalid_argument("ensure_rope_inv_freq_device: head_dim must be positive and even.");
    }
    if (rope_theta <= 0.0f) {
        throw std::invalid_argument("ensure_rope_inv_freq_device: rope_theta must be positive.");
    }
    const int half_dim = head_dim / 2;
    const size_t bytes = static_cast<size_t>(half_dim) * sizeof(float);
    if (impl.rope_inv_freq_half_dim == half_dim &&
        impl.rope_inv_freq_theta == rope_theta &&
        impl.rope_inv_freq.bytes() >= bytes) {
        return;
    }

    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int k = 0; k < half_dim; ++k) {
        const float freq = std::pow(rope_theta, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
        inv_freq[k] = 1.0f / freq;
    }
    ensure_device_buffer(impl.rope_inv_freq, bytes, scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(impl.rope_inv_freq.data(), inv_freq.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync rope_inv_freq");
    impl.rope_inv_freq_half_dim = half_dim;
    impl.rope_inv_freq_theta = rope_theta;
}

AttentionMaskInputs build_attention_mask_inputs(SelfAttnCudaState::Impl& impl,
                                                ForwardScratchBuffers& scratch,
                                                const std::vector<int>& sample_ids,
                                                const std::vector<int>& pad_lens_by_sample,
                                                cudaStream_t stream,
                                                std::uint64_t* scratch_realloc_counter) {
    AttentionMaskInputs mask_inputs{};
    prepare_attention_mask_inputs(impl,
                                  scratch,
                                  sample_ids,
                                  pad_lens_by_sample,
                                  stream,
                                  scratch_realloc_counter,
                                  mask_inputs.d_sample_ids,
                                  mask_inputs.d_pad_ptr,
                                  mask_inputs.pad_size);
    return mask_inputs;
}

template <typename Traits>
void prepare_decode_qkv(SelfAttnCudaState::Impl& impl,
                        ForwardScratchBuffers& scratch,
                        const std::vector<int>& offsets,
                        const SelfAttnCudaParams& params,
                        const ForwardShapeInfo& shape,
                        const ForwardDeviceResources& resources,
                        cudaStream_t stream,
                        std::uint64_t* scratch_realloc_counter) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;

    apply_qkv_projection_biases<Traits>(stream, scratch, shape, resources);

    ensure_device_buffer(scratch.q, static_cast<size_t>(shape.batch) * params.num_heads * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.k, static_cast<size_t>(shape.batch) * params.num_heads_kv * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.v, static_cast<size_t>(shape.batch) * params.num_heads_kv * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);

    const int q_total = shape.batch * params.num_heads * params.head_dim;
    const int kv_total = shape.batch * params.num_heads_kv * params.head_dim;
    const int q_blocks = (q_total + kThreads - 1) / kThreads;
    const int kv_blocks = (kv_total + kThreads - 1) / kThreads;
    split_transpose_seq1_kernel<DeviceType><<<q_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.q_proj.data()),
        static_cast<DeviceType*>(scratch.q.data()),
        shape.batch, params.num_heads, params.head_dim);
    split_transpose_seq1_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.k_proj.data()),
        static_cast<DeviceType*>(scratch.k.data()),
        shape.batch, params.num_heads_kv, params.head_dim);
    split_transpose_seq1_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.v_proj.data()),
        static_cast<DeviceType*>(scratch.v.data()),
        shape.batch, params.num_heads_kv, params.head_dim);
    cuda_check(cudaGetLastError(), "split_transpose_seq1_kernel");
    apply_qk_norm_if_enabled<Traits>(stream, scratch, params, shape, resources);

    upload_offsets_to_device(scratch, offsets, shape.batch, stream, scratch_realloc_counter);
    ensure_rope_inv_freq_device(impl, params.head_dim, params.rope_theta, stream, scratch_realloc_counter);

    const int rope_q_total = shape.batch * params.num_heads * (params.head_dim / 2);
    const int rope_k_total = shape.batch * params.num_heads_kv * (params.head_dim / 2);
    const int rope_q_blocks = (rope_q_total + kThreads - 1) / kThreads;
    const int rope_k_blocks = (rope_k_total + kThreads - 1) / kThreads;
    rope_seq1_kernel<DeviceType><<<rope_q_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(scratch.q.data()),
        static_cast<const int*>(scratch.offsets.data()),
        static_cast<const float*>(impl.rope_inv_freq.data()),
        shape.batch, params.num_heads, params.head_dim);
    rope_seq1_kernel<DeviceType><<<rope_k_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(scratch.k.data()),
        static_cast<const int*>(scratch.offsets.data()),
        static_cast<const float*>(impl.rope_inv_freq.data()),
        shape.batch, params.num_heads_kv, params.head_dim);
    cuda_check(cudaGetLastError(), "rope_seq1_kernel");
}

template <typename Traits>
void prepare_prefill_qkv(SelfAttnCudaState::Impl& impl,
                         ForwardScratchBuffers& scratch,
                         const std::vector<int>& offsets,
                         const SelfAttnCudaParams& params,
                         const ForwardShapeInfo& shape,
                         const ForwardDeviceResources& resources,
                         cudaStream_t stream,
                         std::uint64_t* scratch_realloc_counter) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;

    apply_qkv_projection_biases<Traits>(stream, scratch, shape, resources);

    ensure_device_buffer(scratch.q, static_cast<size_t>(shape.batch) * params.num_heads * shape.seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.k, static_cast<size_t>(shape.batch) * params.num_heads_kv * shape.seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.v, static_cast<size_t>(shape.batch) * params.num_heads_kv * shape.seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);

    const int q_total = shape.batch * params.num_heads * shape.seq_len * params.head_dim;
    const int kv_total = shape.batch * params.num_heads_kv * shape.seq_len * params.head_dim;
    const int q_blocks = (q_total + kThreads - 1) / kThreads;
    const int kv_blocks = (kv_total + kThreads - 1) / kThreads;
    split_transpose_kernel<DeviceType><<<q_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.q_proj.data()),
        static_cast<DeviceType*>(scratch.q.data()),
        shape.batch, shape.seq_len, params.num_heads, params.head_dim);
    split_transpose_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.k_proj.data()),
        static_cast<DeviceType*>(scratch.k.data()),
        shape.batch, shape.seq_len, params.num_heads_kv, params.head_dim);
    split_transpose_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.v_proj.data()),
        static_cast<DeviceType*>(scratch.v.data()),
        shape.batch, shape.seq_len, params.num_heads_kv, params.head_dim);
    cuda_check(cudaGetLastError(), "split_transpose_kernel");
    apply_qk_norm_if_enabled<Traits>(stream, scratch, params, shape, resources);

    upload_offsets_to_device(scratch, offsets, shape.batch, stream, scratch_realloc_counter);
    ensure_rope_inv_freq_device(impl, params.head_dim, params.rope_theta, stream, scratch_realloc_counter);

    const int rope_q_total = shape.batch * params.num_heads * shape.seq_len * (params.head_dim / 2);
    const int rope_k_total = shape.batch * params.num_heads_kv * shape.seq_len * (params.head_dim / 2);
    const int rope_q_blocks = (rope_q_total + kThreads - 1) / kThreads;
    const int rope_k_blocks = (rope_k_total + kThreads - 1) / kThreads;
    rope_kernel<DeviceType><<<rope_q_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(scratch.q.data()),
        static_cast<const int*>(scratch.offsets.data()),
        static_cast<const float*>(impl.rope_inv_freq.data()),
        shape.batch, params.num_heads, shape.seq_len, params.head_dim);
    rope_kernel<DeviceType><<<rope_k_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(scratch.k.data()),
        static_cast<const int*>(scratch.offsets.data()),
        static_cast<const float*>(impl.rope_inv_freq.data()),
        shape.batch, params.num_heads_kv, shape.seq_len, params.head_dim);
    cuda_check(cudaGetLastError(), "rope_kernel");
}

template <typename Traits>
DecodeAttentionResources prepare_decode_attention_resources(SelfAttnCudaState::Impl& impl,
                                                            ForwardScratchBuffers& scratch,
                                                            const std::vector<int>& sample_ids,
                                                            const std::vector<int>& pad_lens_by_sample,
                                                            const SelfAttnCudaParams& params,
                                                            const ForwardShapeInfo& shape,
                                                            bool use_fused_decode_attention,
                                                            const std::vector<int>& active_lens,
                                                            int max_len,
                                                            cudaStream_t stream,
                                                            std::uint64_t* scratch_realloc_counter) {
    DecodeAttentionResources decode_resources{};
    const CacheBatchView cache_view = prepare_cache_batch_view<Traits>(
        impl,
        scratch,
        sample_ids,
        shape.batch,
        params.num_heads_kv,
        max_len,
        params.head_dim,
        use_fused_decode_attention,
        true,
        stream);
    decode_resources.cache_k_ptr = cache_view.cache_k_ptr;
    decode_resources.cache_v_ptr = cache_view.cache_v_ptr;
    decode_resources.score_capacity = cache_view.score_capacity;
    decode_resources.cache_leading_dim = cache_view.cache_leading_dim;
    decode_resources.cache_instance_stride_elems = cache_view.cache_instance_stride_elems;

    if (use_fused_decode_attention) {
        impl.stats.decode_fused_attention_hits += 1;
        if (scratch.scores.bytes() > 0) {
            scratch.scores = DeviceBuffer();
        }
    } else {
        ensure_device_buffer(scratch.scores, static_cast<size_t>(shape.batch) * params.num_heads * decode_resources.score_capacity * Traits::kElementBytes, scratch_realloc_counter);
    }

    if (active_lens.size() != static_cast<size_t>(shape.batch)) {
        throw std::invalid_argument("prepare_decode_attention_resources: active_lens size mismatch.");
    }
    for (int len : active_lens) {
        if (len < 0 || len > decode_resources.score_capacity) {
            throw std::invalid_argument("prepare_decode_attention_resources: active_lens value out of range.");
        }
    }

    ensure_device_buffer(scratch.context, static_cast<size_t>(shape.batch) * params.num_heads * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    const size_t active_len_bytes = static_cast<size_t>(shape.batch) * sizeof(int);
    ensure_device_buffer(scratch.decode_active_len, active_len_bytes, scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(scratch.decode_active_len.data(), active_lens.data(), active_len_bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync decode_active_len");
    decode_resources.d_active_len = static_cast<const int*>(scratch.decode_active_len.data());
    decode_resources.mask_inputs = build_attention_mask_inputs(impl,
                                                               scratch,
                                                               sample_ids,
                                                               pad_lens_by_sample,
                                                               stream,
                                                               scratch_realloc_counter);
    return decode_resources;
}

template <typename Traits>
void launch_decode_attention_pipeline(cublasHandle_t handle,
                                      cudaStream_t stream,
                                      ForwardScratchBuffers& scratch,
                                      const SelfAttnCudaParams& params,
                                      const ForwardShapeInfo& shape,
                                      const ForwardDeviceResources& resources,
                                      const DecodeAttentionResources& decode_resources,
                                      bool use_fused_decode_attention,
                                      std::uint64_t* scratch_realloc_counter) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const float decode_score_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

    if (use_fused_decode_attention) {
        const int fused_threads = pick_fused_threads(params.head_dim, kThreads);
        const size_t fused_shared_bytes = fused_attention_shared_bytes(fused_threads, params.head_dim);
        const int decode_rows = shape.batch * params.num_heads;
        fused_decode_attention_kernel<DeviceType><<<decode_rows, fused_threads, fused_shared_bytes, stream>>>(
            static_cast<const DeviceType*>(scratch.q.data()),
            static_cast<const DeviceType*>(decode_resources.cache_k_ptr),
            static_cast<const DeviceType*>(decode_resources.cache_v_ptr),
            static_cast<DeviceType*>(scratch.context.data()),
            decode_resources.mask_inputs.d_sample_ids,
            decode_resources.mask_inputs.d_pad_ptr,
            decode_resources.mask_inputs.pad_size,
            shape.batch,
            params.num_heads,
            params.num_heads_kv,
            shape.repeat_factor,
            params.head_dim,
            decode_resources.score_capacity,
            decode_resources.d_active_len,
            decode_score_scale);
        cuda_check(cudaGetLastError(), "fused_decode_attention_kernel");
    } else {
        launch_qk_grouped_batched_gemm<Traits>(handle, decode_resources.cache_k_ptr, scratch.q.data(), scratch.scores.data(),
                                               shape.batch, params.num_heads_kv, shape.repeat_factor, 1, decode_resources.score_capacity,
                                               params.head_dim, decode_resources.cache_leading_dim, decode_resources.cache_instance_stride_elems);

        const int decode_rows = shape.batch * params.num_heads;
        mask_scale_softmax_decode_kernel<DeviceType><<<decode_rows, kThreads, kThreads * sizeof(float), stream>>>(
            static_cast<DeviceType*>(scratch.scores.data()),
            decode_resources.mask_inputs.d_sample_ids,
            decode_resources.mask_inputs.d_pad_ptr,
            decode_resources.mask_inputs.pad_size,
            shape.batch, params.num_heads, decode_resources.score_capacity, decode_resources.d_active_len, decode_score_scale);
        cuda_check(cudaGetLastError(), "mask_scale_softmax_decode_kernel");

        launch_av_grouped_batched_gemm<Traits>(handle, scratch.scores.data(), decode_resources.cache_v_ptr, scratch.context.data(),
                                               shape.batch, params.num_heads_kv, shape.repeat_factor, 1, decode_resources.score_capacity,
                                               params.head_dim, decode_resources.cache_leading_dim, decode_resources.cache_instance_stride_elems);
    }

    launch_output_projection<Traits>(handle,
                                     stream,
                                     scratch,
                                     shape.batch,
                                     1,
                                     params.num_heads,
                                     params.head_dim,
                                     shape.rows,
                                     shape.q_out_dim,
                                     shape.o_out_dim,
                                     resources.o_weight_ptr,
                                     resources.d_o_bias,
                                     scratch_realloc_counter,
                                     "merge_heads_kernel decode_seq1");
}

template <typename Traits>
Tensor run_decode_seq1_path(SelfAttnCudaState::Impl& impl,
                            ForwardScratchBuffers& scratch,
                            const std::vector<int>& sample_ids,
                            const std::vector<int>& offsets,
                            const std::vector<int>& pad_lens_by_sample,
                            const SelfAttnCudaParams& params,
                            const ForwardShapeInfo& shape,
                            const ForwardDeviceResources& resources,
                            cudaStream_t stream,
                            cublasHandle_t handle,
                            std::uint64_t* scratch_realloc_counter) {
    const bool use_fused_decode_attention = true;
    prepare_decode_qkv<Traits>(impl,
                               scratch,
                               offsets,
                               params,
                               shape,
                               resources,
                               stream,
                               scratch_realloc_counter);

    const ActiveCacheInfo active_cache = append_kv_cache<Traits>(impl,
                                                                  sample_ids,
                                                                  shape.batch,
                                                                  shape.seq_len,
                                                                  params.num_heads_kv,
                                                                  params.head_dim,
                                                                  scratch.k,
                                                                  scratch.v,
                                                                  stream);
    if (active_cache.max_len <= 0) {
        throw std::runtime_error("self_attn_forward_cuda: cache length is zero after append.");
    }

    const DecodeAttentionResources decode_resources = prepare_decode_attention_resources<Traits>(
        impl,
        scratch,
        sample_ids,
        pad_lens_by_sample,
        params,
        shape,
        use_fused_decode_attention,
        active_cache.active_lens,
        active_cache.max_len,
        stream,
        scratch_realloc_counter);

    auto launch_decode_pipeline = [&]() {
        launch_decode_attention_pipeline<Traits>(handle,
                                                stream,
                                                scratch,
                                                params,
                                                shape,
                                                resources,
                                                decode_resources,
                                                use_fused_decode_attention,
                                                scratch_realloc_counter);
    };

    bool launched = false;
    if (use_fused_decode_attention && shape.batch == 1) {
        DecodeGraphCache& decode_graph = impl.decode_graph;
        const int sample_id = sample_ids[0];
        const bool can_reuse = can_reuse_decode_graph(
            decode_graph,
            shape.batch,
            sample_id,
            params.num_heads,
            params.head_dim,
            shape.q_out_dim,
            shape.q_out_dim,
            shape.k_out_dim,
            shape.v_out_dim,
            shape.o_out_dim,
            shape.repeat_factor,
            decode_resources.score_capacity,
            decode_resources.mask_inputs.pad_size,
            params.use_qk_norm,
            decode_resources.cache_k_ptr,
            decode_resources.cache_v_ptr,
            decode_resources.mask_inputs.d_pad_ptr,
            resources.d_norm_weight,
            resources.d_q_norm_weight,
            resources.d_k_norm_weight,
            resources.q_weight_ptr,
            resources.k_weight_ptr,
            resources.v_weight_ptr,
            resources.o_weight_ptr,
            resources.d_q_bias,
            resources.d_k_bias,
            resources.d_v_bias,
            resources.d_o_bias);
        if (!can_reuse) {
            // Warm up once before graph capture so any lazy allocations happen
            // outside capture (e.g. allocator/cublas internal buffers).
            launch_decode_pipeline();
            cuda_check(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize decode_seq1 warmup");

            decode_graph.reset();
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture decode_seq1");
            launch_decode_pipeline();
            cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture decode_seq1");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                       "cudaGraphInstantiate decode_seq1");
            decode_graph.graph = graph;
            decode_graph.exec = exec;
            fill_decode_graph_signature(
                decode_graph,
                shape.batch,
                sample_id,
                params.num_heads,
                params.head_dim,
                shape.q_out_dim,
                shape.q_out_dim,
                shape.k_out_dim,
                shape.v_out_dim,
                shape.o_out_dim,
                shape.repeat_factor,
                decode_resources.score_capacity,
                decode_resources.mask_inputs.pad_size,
                params.use_qk_norm,
                decode_resources.cache_k_ptr,
                decode_resources.cache_v_ptr,
                decode_resources.mask_inputs.d_pad_ptr,
                resources.d_norm_weight,
                resources.d_q_norm_weight,
                resources.d_k_norm_weight,
                resources.q_weight_ptr,
                resources.k_weight_ptr,
                resources.v_weight_ptr,
                resources.o_weight_ptr,
                resources.d_q_bias,
                resources.d_k_bias,
                resources.d_v_bias,
                resources.d_o_bias);
            impl.stats.decode_graph_captures += 1;
        }
        cuda_check(cudaGraphLaunch(decode_graph.exec, stream), "cudaGraphLaunch decode_seq1");
        impl.stats.decode_graph_launches += 1;
        launched = true;
    }
    if (!launched) {
        launch_decode_pipeline();
    }

    return copy_output_to_host<Traits>(stream,
                                       scratch,
                                       shape.rows,
                                       shape.o_out_dim,
                                       shape.batch,
                                       1,
                                       "cudaMemcpyAsync output decode_seq1",
                                       "cudaStreamSynchronize self_attn_forward_cuda decode_seq1");
}

template <typename Traits>
Tensor run_prefill_path(SelfAttnCudaState::Impl& impl,
                        ForwardScratchBuffers& scratch,
                        const std::vector<int>& sample_ids,
                        const std::vector<int>& offsets,
                        const std::vector<int>& pad_lens_by_sample,
                        const SelfAttnCudaParams& params,
                        const ForwardShapeInfo& shape,
                        const ForwardDeviceResources& resources,
                        cudaStream_t stream,
                        cublasHandle_t handle,
                        std::uint64_t* scratch_realloc_counter) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const bool use_fused_prefill_attention = true;

    prepare_prefill_qkv<Traits>(impl,
                                scratch,
                                offsets,
                                params,
                                shape,
                                resources,
                                stream,
                                scratch_realloc_counter);

    const ActiveCacheInfo active_cache = append_kv_cache<Traits>(impl,
                                                                  sample_ids,
                                                                  shape.batch,
                                                                  shape.seq_len,
                                                                  params.num_heads_kv,
                                                                  params.head_dim,
                                                                  scratch.k,
                                                                  scratch.v,
                                                                  stream);
    const int total_len = active_cache.max_len;
    if (total_len <= 0) {
        throw std::runtime_error("self_attn_forward_cuda: cache length is zero after append.");
    }
    for (int len : active_cache.active_lens) {
        if (len != total_len) {
            throw std::invalid_argument("self_attn_forward_cuda: prefill path requires uniform active cache lengths.");
        }
    }

    const CacheBatchView cache_view = prepare_cache_batch_view<Traits>(
        impl,
        scratch,
        sample_ids,
        shape.batch,
        params.num_heads_kv,
        total_len,
        params.head_dim,
        use_fused_prefill_attention,
        false,
        stream);
    const void* cache_k_ptr = cache_view.cache_k_ptr;
    const void* cache_v_ptr = cache_view.cache_v_ptr;
    const int cache_leading_dim = cache_view.cache_leading_dim;
    const long long cache_instance_stride_elems = cache_view.cache_instance_stride_elems;

    const AttentionMaskInputs mask_inputs = build_attention_mask_inputs(impl,
                                                                        scratch,
                                                                        sample_ids,
                                                                        pad_lens_by_sample,
                                                                        stream,
                                                                        scratch_realloc_counter);

    ensure_device_buffer(scratch.context, static_cast<size_t>(shape.batch) * params.num_heads * shape.seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    const float score_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    if (use_fused_prefill_attention) {
        impl.stats.prefill_fused_attention_hits += 1;
        if (scratch.scores.bytes() > 0) {
            scratch.scores = DeviceBuffer();
        }

        const int fused_threads = pick_fused_threads(params.head_dim, kThreads);
        const size_t fused_shared_bytes = fused_attention_shared_bytes(fused_threads, params.head_dim);
        const int prefill_rows = shape.batch * params.num_heads * shape.seq_len;
        fused_prefill_attention_kernel<DeviceType><<<prefill_rows, fused_threads, fused_shared_bytes, stream>>>(
            static_cast<const DeviceType*>(scratch.q.data()),
            static_cast<const DeviceType*>(cache_k_ptr),
            static_cast<const DeviceType*>(cache_v_ptr),
            static_cast<DeviceType*>(scratch.context.data()),
            mask_inputs.d_sample_ids,
            mask_inputs.d_pad_ptr,
            mask_inputs.pad_size,
            shape.batch,
            params.num_heads,
            params.num_heads_kv,
            shape.repeat_factor,
            shape.seq_len,
            total_len,
            params.head_dim,
            score_scale);
        cuda_check(cudaGetLastError(), "fused_prefill_attention_kernel");
    } else {
        ensure_device_buffer(scratch.scores, static_cast<size_t>(shape.batch) * params.num_heads * shape.seq_len * total_len * Traits::kElementBytes, scratch_realloc_counter);
        launch_qk_grouped_batched_gemm<Traits>(handle, cache_k_ptr, scratch.q.data(), scratch.scores.data(),
                                               shape.batch, params.num_heads_kv, shape.repeat_factor, shape.seq_len, total_len, params.head_dim,
                                               cache_leading_dim, cache_instance_stride_elems);

        const int row_count = shape.batch * params.num_heads * shape.seq_len;
        mask_scale_softmax_kernel<DeviceType><<<row_count, kThreads, kThreads * sizeof(float), stream>>>(
            static_cast<DeviceType*>(scratch.scores.data()),
            mask_inputs.d_sample_ids,
            mask_inputs.d_pad_ptr,
            mask_inputs.pad_size,
            shape.batch, params.num_heads, shape.seq_len, total_len, score_scale);
        cuda_check(cudaGetLastError(), "mask_scale_softmax_kernel");

        launch_av_grouped_batched_gemm<Traits>(handle, scratch.scores.data(), cache_v_ptr, scratch.context.data(),
                                               shape.batch, params.num_heads_kv, shape.repeat_factor, shape.seq_len, total_len, params.head_dim,
                                               cache_leading_dim, cache_instance_stride_elems);
    }

    launch_output_projection<Traits>(handle,
                                     stream,
                                     scratch,
                                     shape.batch,
                                     shape.seq_len,
                                     params.num_heads,
                                     params.head_dim,
                                     shape.rows,
                                     shape.q_out_dim,
                                     shape.o_out_dim,
                                     resources.o_weight_ptr,
                                     resources.d_o_bias,
                                     scratch_realloc_counter,
                                     "merge_heads_kernel");
    return copy_output_to_host<Traits>(stream,
                                       scratch,
                                       shape.rows,
                                       shape.o_out_dim,
                                       shape.batch,
                                       shape.seq_len,
                                       "cudaMemcpyAsync output",
                                       "cudaStreamSynchronize self_attn_forward_cuda");
}

} // namespace

Tensor self_attn_forward_cuda(const Tensor& input,
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
                              SelfAttnCudaState& state) {
    using Traits = CudaPrecisionTraits<data_type>;
    SelfAttnCudaState::Impl& impl = *state.impl_;
    const ForwardShapeInfo shape = validate_forward_shape_and_state(input,
                                                                    sample_ids,
                                                                    offsets,
                                                                    params,
                                                                    norm_weight,
                                                                    q_norm_weight,
                                                                    k_norm_weight,
                                                                    q_weight,
                                                                    q_bias,
                                                                    k_weight,
                                                                    k_bias,
                                                                    v_weight,
                                                                    v_bias,
                                                                    o_weight,
                                                                    o_bias,
                                                                    impl);

    auto& ctx = get_context();
    if (!ctx.available()) {
        throw std::runtime_error("self_attn_forward_cuda: CUDA runtime unavailable.");
    }

    std::lock_guard<std::mutex> lock(ctx.mutex());
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.handle();

    auto& scratch = impl.scratch;
    std::uint64_t* scratch_realloc_counter = &impl.stats.scratch_reallocations;
    const ForwardDeviceResources resources = prepare_forward_device_resources<Traits>(
        ctx,
        impl,
        scratch,
        params,
        shape,
        input,
        norm_weight,
        q_norm_weight,
        k_norm_weight,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        stream,
        handle,
        scratch_realloc_counter);

    if (shape.seq_len == 1) {
        impl.stats.decode_seq1_path_hits += 1;
        return run_decode_seq1_path<Traits>(impl,
                                            scratch,
                                            sample_ids,
                                            offsets,
                                            pad_lens_by_sample,
                                            params,
                                            shape,
                                            resources,
                                            stream,
                                            handle,
                                            scratch_realloc_counter);
    }

    return run_prefill_path<Traits>(impl,
                                    scratch,
                                    sample_ids,
                                    offsets,
                                    pad_lens_by_sample,
                                    params,
                                    shape,
                                    resources,
                                    stream,
                                    handle,
                                    scratch_realloc_counter);
}

} // namespace ops
} // namespace cuda
} // namespace easy_llm
