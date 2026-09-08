#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// PagedAttention decode kernel, following vLLM's own paged_attention_v1
// design: one thread block per (sequence, query_head), split into NUM_WARPS
// warps that each own a disjoint, strided subset of the sequence's KV
// *blocks* (not individual tokens -- a block is the physical storage unit,
// so a warp that owns a block iterates its block_size tokens sequentially,
// same access pattern the allocator already gives contiguous). Each warp
// runs its own independent streaming/online-softmax over just its assigned
// blocks, in parallel with every other warp; within a warp, head_dim is
// split across the 32 lanes and each per-token dot product is reduced via
// __shfl_down_sync (no shared memory, no barrier -- warp-synchronous).
// After every warp finishes its local partial softmax, one __syncthreads()
// (the only one in the whole kernel) lets a final merge combine the
// per-warp (max, sum, acc) triples with the standard online-softmax-merge
// algebra: for two partial softmaxes over disjoint token sets,
//   merged_max = max(max_a, max_b)
//   merged_sum = sum_a*exp(max_a-merged_max) + sum_b*exp(max_b-merged_max)
//   merged_acc = acc_a*exp(max_a-merged_max) + acc_b*exp(max_b-merged_max)
// which generalizes to N partial results by folding pairwise.
constexpr int MAX_ELEMS_PER_LANE = 8;  // supports head_dim up to 32*8 = 256
constexpr int NUM_WARPS = 4;

__global__ void paged_attention_decode_kernel(
    const float* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ v_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float* __restrict__ out,               // [num_seqs, num_heads, head_dim]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / num_queries_per_kv;
    int context_len = context_lens[seq_idx];
    int num_context_blocks = (context_len + block_size - 1) / block_size;

    const float* q_ptr = q + (seq_idx * num_heads + head_idx) * head_dim;

    int elems = (head_dim + 31) / 32;  // elements this lane owns, at indices lane, lane+32, lane+64, ...
    float q_val[MAX_ELEMS_PER_LANE];
    float acc[MAX_ELEMS_PER_LANE];
#pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_LANE; e++) {
        int d = lane + e * 32;
        q_val[e] = (e < elems && d < head_dim) ? q_ptr[d] : 0.0f;
        acc[e] = 0.0f;
    }

    const int* block_table = block_tables + seq_idx * max_blocks_per_seq;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Each warp strides across the sequence's KV blocks, block_size tokens
    // (contiguous in physical storage) at a time.
    for (int block_idx = warp_id; block_idx < num_context_blocks; block_idx += NUM_WARPS) {
        int block_id = block_table[block_idx];
        int tokens_in_block = min(block_size, context_len - block_idx * block_size);

        for (int slot = 0; slot < tokens_in_block; slot++) {
            const float* k_ptr =
                k_cache + ((static_cast<long>(block_id) * block_size + slot) * num_kv_heads + kv_head_idx) * head_dim;
            const float* v_ptr =
                v_cache + ((static_cast<long>(block_id) * block_size + slot) * num_kv_heads + kv_head_idx) * head_dim;

            float partial = 0.0f;
#pragma unroll
            for (int e = 0; e < MAX_ELEMS_PER_LANE; e++) {
                if (e < elems) {
                    int d = lane + e * 32;
                    if (d < head_dim) partial += q_val[e] * k_ptr[d];
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, offset);
            }
            float score = __shfl_sync(0xffffffffu, partial, 0) * scale;

            float new_max = fmaxf(running_max, score);
            float correction = __expf(running_max - new_max);
            float p = __expf(score - new_max);
            running_sum = running_sum * correction + p;
#pragma unroll
            for (int e = 0; e < MAX_ELEMS_PER_LANE; e++) {
                if (e < elems) {
                    int d = lane + e * 32;
                    float v_val = (d < head_dim) ? v_ptr[d] : 0.0f;
                    acc[e] = acc[e] * correction + p * v_val;
                }
            }
            running_max = new_max;
        }
    }

    // Cross-warp online-softmax merge. Layout of dynamic shared memory:
    // [0, NUM_WARPS*head_dim)                 -- each warp's acc, per lane's owned indices
    // [NUM_WARPS*head_dim, +NUM_WARPS)        -- each warp's running_max
    // [NUM_WARPS*head_dim+NUM_WARPS, +NUM_WARPS) -- each warp's running_sum
    extern __shared__ float smem[];
    float* s_acc = smem;
    float* s_max = smem + NUM_WARPS * head_dim;
    float* s_sum = s_max + NUM_WARPS;

#pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_LANE; e++) {
        if (e < elems) {
            int d = lane + e * 32;
            if (d < head_dim) s_acc[warp_id * head_dim + d] = acc[e];
        }
    }
    if (lane == 0) {
        s_max[warp_id] = running_max;
        s_sum[warp_id] = running_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float merged_max = -1e30f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) merged_max = fmaxf(merged_max, s_max[w]);

        float merged_sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) merged_sum += s_sum[w] * __expf(s_max[w] - merged_max);

#pragma unroll
        for (int e = 0; e < MAX_ELEMS_PER_LANE; e++) {
            if (e < elems) {
                int d = lane + e * 32;
                if (d < head_dim) {
                    float merged_acc = 0.0f;
#pragma unroll
                    for (int w = 0; w < NUM_WARPS; w++) {
                        merged_acc += s_acc[w * head_dim + d] * __expf(s_max[w] - merged_max);
                    }
                    out[(seq_idx * num_heads + head_idx) * head_dim + d] = merged_acc / merged_sum;
                }
            }
        }
    }
}

torch::Tensor paged_attention_decode(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(block_tables.dtype() == torch::kInt32, "block_tables must be int32");
    TORCH_CHECK(context_lens.dtype() == torch::kInt32, "context_lens must be int32");

    q = q.contiguous();
    k_cache = k_cache.contiguous();
    v_cache = v_cache.contiguous();
    block_tables = block_tables.contiguous();
    context_lens = context_lens.contiguous();

    int num_seqs = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int num_kv_heads = k_cache.size(2);
    int block_size = k_cache.size(1);
    int max_blocks_per_seq = block_tables.size(1);

    TORCH_CHECK(head_dim <= 32 * MAX_ELEMS_PER_LANE, "head_dim too large for this kernel's per-lane register budget");

    auto out = torch::empty_like(q);

    dim3 grid(num_seqs, num_heads);
    dim3 block(32 * NUM_WARPS);
    size_t shared_mem = (NUM_WARPS * head_dim + 2 * NUM_WARPS) * sizeof(float);

    paged_attention_decode_kernel<<<grid, block, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        out.data_ptr<float>(),
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        static_cast<float>(scale));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_decode", &paged_attention_decode, "Paged attention decode step (CUDA)");
}
