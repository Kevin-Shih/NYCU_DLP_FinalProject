/*!
**************************************************************************************************
* BoxeR
* Copyright (c) 2022. All Rights Reserved.
* Licensed under the MIT License [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/fundamentalvision/Deformable-DETR
**************************************************************************************************
*/
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < n;                                          \
        i += blockDim.x * gridDim.x)


const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
    return (N + num_threads - 1) / num_threads;
}


template <typename scalar_t>
__device__ scalar_t instance_attn_im2col_bilinear(
    const scalar_t* &bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const scalar_t &h,
    const scalar_t &w,
    const int &m,
    const int &c
)
{
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh;
    const scalar_t hw = 1 - lw;

    const int w_stride = nheads * channels;
    const int h_stride = w_stride * width;
    const int h_low_ptr_offset = h_low * h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int value_ptr = m * channels + c;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + value_ptr;
        v1 = bottom_data[ptr1];
    }

    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + value_ptr;
        v2 = bottom_data[ptr2];
    }

    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + value_ptr;
        v3 = bottom_data[ptr3];
    }

    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + value_ptr;
        v4 = bottom_data[ptr4];
    }

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}


template <typename scalar_t>
__device__ void instance_attn_col2im_bilinear(
    const scalar_t* &bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const scalar_t &h,
    const scalar_t &w,
    const int &m,
    const int &c,
    const scalar_t &top_grad,
    const scalar_t &top_mask_grad,
    const scalar_t &spatial_attn_weight,
    const scalar_t &level_attn_weight,
    scalar_t* &grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh;
    const scalar_t hw = 1 - lw;

    const int w_stride = nheads * channels;
    const int h_stride = w_stride * width;
    const int h_low_ptr_offset = h_low * h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int value_ptr = m * channels + c;

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    // const scalar_t top_grad_value = top_grad * spatial_attn_weight;
    const scalar_t top_grad_value = top_grad * spatial_attn_weight + top_mask_grad * level_attn_weight;
    scalar_t grad_h_weight = 0, grad_w_weight = 0;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + value_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    }

    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + value_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    }

    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + value_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    }

    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + value_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    }

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    *grad_spatial_attn_weight = top_grad * val;
    *grad_level_attn_weight = top_mask_grad * val;
    *grad_sampling_loc = width * grad_w_weight * top_grad_value;
    *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}


template <typename scalar_t>
__device__ void instance_attn_col2im_bilinear_gm(
    const scalar_t* &bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const scalar_t &h,
    const scalar_t &w,
    const int &m,
    const int &c,
    const scalar_t &top_grad,
    const scalar_t &top_mask_grad,
    const scalar_t &spatial_attn_weight,
    const scalar_t &level_attn_weight,
    scalar_t* &grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh;
    const scalar_t hw = 1 - lw;

    const int w_stride = nheads * channels;
    const int h_stride = w_stride * width;
    const int h_low_ptr_offset = h_low * h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int value_ptr = m * channels + c;

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    // const scalar_t top_grad_value = top_grad * spatial_attn_weight;
    const scalar_t top_grad_value = top_grad * spatial_attn_weight + top_mask_grad * level_attn_weight;
    scalar_t grad_h_weight = 0, grad_w_weight = 0;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + value_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    }

    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + value_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    }

    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + value_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    }

    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + value_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    }

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    atomicAdd(grad_spatial_attn_weight, top_grad * val);
    atomicAdd(grad_level_attn_weight, top_mask_grad * val);
    atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
    atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}


template <typename scalar_t>
__global__ void instance_attn_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col,
    scalar_t* data_mask_col
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        scalar_t* data_col_ptr = data_col + index;
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;
        scalar_t* data_mask_col_ptr = data_mask_col + data_mask_ptr_init_offset;
        scalar_t col = 0;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const scalar_t* data_value_ptr = data_value + data_value_ptr_init_offset + level_start_id * qid_stride;

            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    const scalar_t val = instance_attn_im2col_bilinear(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col
                    );
                    col += val * spatial_weight;
                    atomicAdd(data_mask_col_ptr + p_col * qid_stride + m_col * channels + c_col, val * level_weight);
                }

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
            }
        }
        *data_col_ptr = col;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
        __shared__ scalar_t cache_grad_spatial_attn_weight[blockSize];
        __shared__ scalar_t cache_grad_level_attn_weight[blockSize];

        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_spatial_attn_weight += grad_sampling_ptr;
        grad_level_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const int data_mask_pos = data_mask_ptr_init_offset + p_col * qid_stride + m_col * channels + c_col;
                const scalar_t top_mask_grad = grad_mask_col[data_mask_pos];

                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + 1 + (threadIdx.x << 1)) = 0;
                *(cache_grad_spatial_attn_weight + threadIdx.x) = 0;
                *(cache_grad_level_attn_weight + threadIdx.x) = 0;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    instance_attn_col2im_bilinear(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col,
                        top_grad,
                        top_mask_grad,
                        spatial_weight,
                        level_weight,
                        grad_value_ptr,
                        cache_grad_sampling_loc + (threadIdx.x << 1),
                        cache_grad_spatial_attn_weight + threadIdx.x,
                        cache_grad_level_attn_weight + threadIdx.x
                    );
                }

                __syncthreads();
                if (tid == 0)
                {
                    scalar_t _grad_w = cache_grad_sampling_loc[0];
                    scalar_t _grad_h = cache_grad_sampling_loc[1];
                    scalar_t _grad_s = cache_grad_spatial_attn_weight[0];
                    scalar_t _grad_l = cache_grad_level_attn_weight[0];

                    int sid = 2;
                    for (unsigned int tid = 1; tid < blockSize; ++tid)
                    {
                        _grad_w += cache_grad_sampling_loc[sid];
                        _grad_h += cache_grad_sampling_loc[sid + 1];
                        _grad_s += cache_grad_spatial_attn_weight[tid];
                        _grad_l += cache_grad_level_attn_weight[tid];
                        sid += 2;
                    }

                    *grad_sampling_loc = _grad_w;
                    *(grad_sampling_loc + 1) = _grad_h;
                    *grad_spatial_attn_weight = _grad_s;
                    *grad_level_attn_weight = _grad_l;
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_spatial_attn_weight += grad_weight_stride;
                grad_level_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
        __shared__ scalar_t cache_grad_spatial_attn_weight[blockSize];
        __shared__ scalar_t cache_grad_level_attn_weight[blockSize];

        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_spatial_attn_weight += grad_sampling_ptr;
        grad_level_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const int data_mask_pos = data_mask_ptr_init_offset + p_col * qid_stride + m_col * channels + c_col;
                const scalar_t top_mask_grad = grad_mask_col[data_mask_pos];

                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + 1 + (threadIdx.x << 1)) = 0;
                *(cache_grad_spatial_attn_weight + threadIdx.x) = 0;
                *(cache_grad_level_attn_weight + threadIdx.x) = 0;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    instance_attn_col2im_bilinear(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col,
                        top_grad,
                        top_mask_grad,
                        spatial_weight,
                        level_weight,
                        grad_value_ptr,
                        cache_grad_sampling_loc + (threadIdx.x << 1),
                        cache_grad_spatial_attn_weight + threadIdx.x,
                        cache_grad_level_attn_weight + threadIdx.x
                    );
                }

                __syncthreads();
                for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_spatial_attn_weight[tid] += cache_grad_spatial_attn_weight[tid + s];
                        cache_grad_level_attn_weight[tid] += cache_grad_level_attn_weight[tid + s];
                        cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
                        cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *grad_sampling_loc = cache_grad_sampling_loc[0];
                    *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
                    *grad_spatial_attn_weight = cache_grad_spatial_attn_weight[0];
                    *grad_level_attn_weight = cache_grad_level_attn_weight[0];
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_spatial_attn_weight += grad_weight_stride;
                grad_level_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}


template <typename scalar_t>
__global__ void instance_attn_col2im_gpu_kernel_shm_reduce_v1(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int _s[];
        scalar_t* cache_grad_sampling_loc = (scalar_t*) _s;
        scalar_t* cache_grad_spatial_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
        scalar_t* cache_grad_level_attn_weight = cache_grad_spatial_attn_weight + blockDim.x;

        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_spatial_attn_weight += grad_sampling_ptr;
        grad_level_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const int data_mask_pos = data_mask_ptr_init_offset + p_col * qid_stride + m_col * channels + c_col;
                const scalar_t top_mask_grad = grad_mask_col[data_mask_pos];

                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + 1 + (threadIdx.x << 1)) = 0;
                *(cache_grad_spatial_attn_weight + threadIdx.x) = 0;
                *(cache_grad_level_attn_weight + threadIdx.x) = 0;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    instance_attn_col2im_bilinear(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col,
                        top_grad,
                        top_mask_grad,
                        spatial_weight,
                        level_weight,
                        grad_value_ptr,
                        cache_grad_sampling_loc + (threadIdx.x << 1),
                        cache_grad_spatial_attn_weight + threadIdx.x,
                        cache_grad_level_attn_weight + threadIdx.x
                    );
                }

                __syncthreads();
                if (tid == 0)
                {
                    scalar_t _grad_w = cache_grad_sampling_loc[0];
                    scalar_t _grad_h = cache_grad_sampling_loc[1];
                    scalar_t _grad_s = cache_grad_spatial_attn_weight[0];
                    scalar_t _grad_l = cache_grad_level_attn_weight[0];

                    int sid = 2;
                    for (unsigned int tid = 1; tid < blockDim.x; ++tid)
                    {
                        _grad_w += cache_grad_sampling_loc[sid];
                        _grad_h += cache_grad_sampling_loc[sid + 1];
                        _grad_s += cache_grad_spatial_attn_weight[tid];
                        _grad_l += cache_grad_level_attn_weight[tid];
                        sid += 2;
                    }

                    *grad_sampling_loc = _grad_w;
                    *(grad_sampling_loc + 1) = _grad_h;
                    *grad_spatial_attn_weight = _grad_s;
                    *grad_level_attn_weight = _grad_l;
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_spatial_attn_weight += grad_weight_stride;
                grad_level_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}


template <typename scalar_t>
__global__ void instance_attn_col2im_gpu_kernel_shm_reduce_v2(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int _s[];
        scalar_t* cache_grad_sampling_loc = (scalar_t*) _s;
        scalar_t* cache_grad_spatial_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
        scalar_t* cache_grad_level_attn_weight = cache_grad_spatial_attn_weight + blockDim.x;

        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_spatial_attn_weight += grad_sampling_ptr;
        grad_level_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;
            
            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const int data_mask_pos = data_mask_ptr_init_offset + p_col * qid_stride + m_col * channels + c_col;
                const scalar_t top_mask_grad = grad_mask_col[data_mask_pos];

                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + 1 + (threadIdx.x << 1)) = 0;
                *(cache_grad_spatial_attn_weight + threadIdx.x) = 0;
                *(cache_grad_level_attn_weight + threadIdx.x) = 0;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    instance_attn_col2im_bilinear(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col,
                        top_grad,
                        top_mask_grad,
                        spatial_weight,
                        level_weight,
                        grad_value_ptr,
                        cache_grad_sampling_loc + (threadIdx.x << 1),
                        cache_grad_spatial_attn_weight + threadIdx.x,
                        cache_grad_level_attn_weight + threadIdx.x
                    );
                }

                __syncthreads();
                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_spatial_attn_weight[tid] += cache_grad_spatial_attn_weight[tid + s];
                        cache_grad_level_attn_weight[tid] += cache_grad_level_attn_weight[tid + s];
                        cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
                        cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];

                        if (tid + (s << 1) < spre)
                        {
                            cache_grad_spatial_attn_weight[tid] += cache_grad_spatial_attn_weight[tid + (s << 1)];
                            cache_grad_level_attn_weight[tid] += cache_grad_level_attn_weight[tid + (s << 1)];
                            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
                            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *grad_sampling_loc = cache_grad_sampling_loc[0];
                    *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
                    *grad_spatial_attn_weight = cache_grad_spatial_attn_weight[0];
                    *grad_level_attn_weight = cache_grad_level_attn_weight[0];
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_spatial_attn_weight += grad_weight_stride;
                grad_level_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}


template <typename scalar_t>
__global__ void instance_attn_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int _s[];
        scalar_t* cache_grad_sampling_loc = (scalar_t*) _s;
        scalar_t* cache_grad_spatial_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
        scalar_t* cache_grad_level_attn_weight = cache_grad_spatial_attn_weight + blockDim.x;

        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_spatial_attn_weight += grad_sampling_ptr;
        grad_level_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const int data_mask_pos = data_mask_ptr_init_offset + p_col * qid_stride + m_col * channels + c_col;
                const scalar_t top_mask_grad = grad_mask_col[data_mask_pos];

                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + 1 + (threadIdx.x << 1)) = 0;
                *(cache_grad_spatial_attn_weight + threadIdx.x) = 0;
                *(cache_grad_level_attn_weight + threadIdx.x) = 0;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    instance_attn_col2im_bilinear(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col,
                        top_grad,
                        top_mask_grad,
                        spatial_weight,
                        level_weight,
                        grad_value_ptr,
                        cache_grad_sampling_loc + (threadIdx.x << 1),
                        cache_grad_spatial_attn_weight + threadIdx.x,
                        cache_grad_level_attn_weight + threadIdx.x
                    );
                }

                __syncthreads();
                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_spatial_attn_weight[tid] += cache_grad_spatial_attn_weight[tid + s];
                        cache_grad_level_attn_weight[tid] += cache_grad_level_attn_weight[tid + s];
                        cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
                        cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];

                        if (tid + (s << 1) < spre)
                        {
                            cache_grad_spatial_attn_weight[tid] += cache_grad_spatial_attn_weight[tid + (s << 1)];
                            cache_grad_level_attn_weight[tid] += cache_grad_level_attn_weight[tid + (s << 1)];
                            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
                            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
                    atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
                    atomicAdd(grad_spatial_attn_weight, cache_grad_spatial_attn_weight[0]);
                    atomicAdd(grad_level_attn_weight, cache_grad_level_attn_weight[0]);
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_spatial_attn_weight += grad_weight_stride;
                grad_level_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}


template <typename scalar_t>
__global__ void instance_attn_col2im_gpu_kernel_gm(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_spatial_attn_weight += grad_sampling_ptr;
        grad_level_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const int data_mask_ptr_init_offset = (b_col * num_query + q_col) * num_point * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col)
            {
                const int data_mask_pos = data_mask_ptr_init_offset + p_col * qid_stride + m_col * channels + c_col;
                const scalar_t top_mask_grad = grad_mask_col[data_mask_pos];

                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t spatial_weight = data_spatial_attn_weight[data_weight_ptr];
                const scalar_t level_weight = data_level_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
                {
                    instance_attn_col2im_bilinear_gm(
                        data_value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        m_col,
                        c_col,
                        top_grad,
                        top_mask_grad,
                        spatial_weight,
                        level_weight,
                        grad_value_ptr,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                    );
                }

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_spatial_attn_weight += grad_weight_stride;
                grad_level_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}


template <typename scalar_t>
void instance_attn_im2col_cuda(
    cudaStream_t stream,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col,
    scalar_t* data_mask_col
)
{
    const int num_kenels = batch_size * num_query * num_heads * channels;
    const int num_actual_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = CUDA_NUM_THREADS;

    instance_attn_im2col_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
            num_kenels,
            data_value,
            data_spatial_shapes,
            data_level_start_index,
            data_sampling_loc,
            data_spatial_attn_weight,
            data_level_attn_weight,
            batch_size,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            data_col,
            data_mask_col
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in instance_attn_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}


template <typename scalar_t>
void instance_attn_col2im_cuda(
    cudaStream_t stream,
    const scalar_t* grad_col,
    const scalar_t* grad_mask_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_spatial_attn_weight,
    const scalar_t* data_level_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_spatial_attn_weight,
    scalar_t* grad_level_attn_weight
)
{
    const int num_threads = (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_actual_kernels = batch_size * num_query * num_heads * channels;

    if (channels > 1024)
    {
        if ((channels & 1023) == 0)
        {
            instance_attn_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 
                num_threads * 4 * sizeof(scalar_t), stream>>>(
                    num_kernels,
                    grad_col,
                    grad_mask_col,
                    data_value,
                    data_spatial_shapes,
                    data_level_start_index,
                    data_sampling_loc,
                    data_spatial_attn_weight,
                    data_level_attn_weight,
                    batch_size,
                    spatial_size,
                    num_heads,
                    channels,
                    num_levels,
                    num_query,
                    num_point,
                    grad_value,
                    grad_sampling_loc,
                    grad_spatial_attn_weight,
                    grad_level_attn_weight
            );
        }
        else
        {
            instance_attn_col2im_gpu_kernel_gm<scalar_t>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels,
                    grad_col,
                    grad_mask_col,
                    data_value,
                    data_spatial_shapes,
                    data_level_start_index,
                    data_sampling_loc,
                    data_spatial_attn_weight,
                    data_level_attn_weight,
                    batch_size,
                    spatial_size,
                    num_heads,
                    channels,
                    num_levels,
                    num_query,
                    num_point,
                    grad_value,
                    grad_sampling_loc,
                    grad_spatial_attn_weight,
                    grad_level_attn_weight
            );
        }
    }
    else
    {
        switch (channels)
        {
            case 1:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 2:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 4:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 8:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 16:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 32:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 64:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 128:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 256:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 512:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            case 1024:
                instance_attn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                        num_kernels,
                        grad_col,
                        grad_mask_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_spatial_attn_weight,
                        data_level_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_spatial_attn_weight,
                        grad_level_attn_weight
                );
                break;
            default:
                if (channels < 64)
                {
                    instance_attn_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
                        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 
                        num_threads * 4 * sizeof(scalar_t), stream>>>(
                            num_kernels,
                            grad_col,
                            grad_mask_col,
                            data_value,
                            data_spatial_shapes,
                            data_level_start_index,
                            data_sampling_loc,
                            data_spatial_attn_weight,
                            data_level_attn_weight,
                            batch_size,
                            spatial_size,
                            num_heads,
                            channels,
                            num_levels,
                            num_query,
                            num_point,
                            grad_value,
                            grad_sampling_loc,
                            grad_spatial_attn_weight,
                            grad_level_attn_weight
                    );
                }
                else
                {
                    instance_attn_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
                        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 
                        num_threads * 4 * sizeof(scalar_t), stream>>>(
                            num_kernels,
                            grad_col,
                            grad_mask_col,
                            data_value,
                            data_spatial_shapes,
                            data_level_start_index,
                            data_sampling_loc,
                            data_spatial_attn_weight,
                            data_level_attn_weight,
                            batch_size,
                            spatial_size,
                            num_heads,
                            channels,
                            num_levels,
                            num_query,
                            num_point,
                            grad_value,
                            grad_sampling_loc,
                            grad_spatial_attn_weight,
                            grad_level_attn_weight
                    );
                }
                break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in instance_attn_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}
