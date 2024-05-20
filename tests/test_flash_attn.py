import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from vllm_flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from vllm_flash_attn.flash_attn_interface import _get_block_size_n

MAX_HEADDIM_SM8x = 192

def maybe_fp8_ref(tensor, kv_cache_dtype, dtype):
    return tensor.to(kv_cache_dtype).to(dtype) if tensor is not None and kv_cache_dtype is not None else tensor

def maybe_fp8(tensor, kv_cache_dtype):
    return tensor.to(kv_cache_dtype).view(dtype=torch.uint8) if tensor is not None and kv_cache_dtype is not None else tensor


# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        output = input[indices]
        # We don't want to reshape input (b ... -> b (...)) since it could change the channel_last
        # memory format to channel_first. In other words, input might not be contiguous.
        # If we don't detach, Pytorch complains about output being a view and is being modified inplace
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        # grad_input[indices] += grad_output
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    """
    Supports concatenating short samples in one sequence. The attention_mask_in_length is utilized to mask other short samples. It helps efficient training of variant lengths-based samples (e.g., the supervised fine-tuning task in large language model).
    The motivation for this function is explained [here](https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286).
    
    For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
        ```
        [
          [2, 3, 0, 0, 0, 0],
          [3, 2, 0, 0, 0, 0],
          [6, 0, 0, 0, 0, 0]
        ]
        ```
    , which refers to the 3D-attention mask:
        ```
        [
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
          ]
        ]
        ```.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = torch.arange(seqlen, device=length.device, dtype=length.dtype).expand(len(length), seqlen) < length.unsqueeze(1)
    real_indices_idx = torch.nonzero(attention_mask_in_length.flatten(), as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch

import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            rotary_dim,
            seqlen_ro,
            output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output

# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            apply_rotary(
                qk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
            apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            apply_rotary(
                dqk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            apply_rotary(
                dq, cos, sin, seqlen_offsets, interleaved=ctx.interleaved, inplace=True, conjugate=True
            )
            apply_rotary(
                dk,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        return dqkv, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets)


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cos, sin, interleaved=False, seqlen_offsets: Union[int, torch.Tensor] = 0):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        apply_rotary(
            k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv

is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(
    q,
    kv,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device="cuda") < sparsity
    return mask


def attention_blocksparse_ref(qkv, blockmask, attn_mask, dropout_p, dropout_mask):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        blockmask: (seqlen / 16, seqlen / 256)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    d = qkv.shape[-1]
    seqlen = qkv.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    blockmask = repeat(blockmask, "s_16 s_256 -> (s_16 16) (s_256 256)")
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(rearrange(~attn_mask, "b s -> b 1 s 1"), 0.0)
    attention = attention.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), 0.0)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1 1"), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128])
# @pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize('seqlen', [128, 256, 384, 512, 768, 1024, 2048])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 384, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize("seqlen", [512])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# @pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_qkvpacked(seqlen, d, dropout_p, causal, local, alibi, deterministic, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen, seqlen, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    out, lse, S_dmask = flash_attn_qkvpacked_func(
        qkv,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        attn = normalize_flash_attn_S(
            attn_unnorm,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            None,
            None,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask, None, None, causal=causal, window_size=window_size
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv, None, attn_bias, dropout_p, dropout_mask, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_qkvpacked_ref(
        qkv,
        None,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    # v = qkv[:, :, 2].float()
    # qk = torch.einsum('bshd,bthd->bhst', qkv[:, :, 0], qkv[:, :, 1]).float()
    # if causal:
    #     causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
    #     qk.masked_fill_(causal_mask, float('-inf'))
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # p_tmp = torch.softmax(qk / math.sqrt(d), -1)
    # p_dropped = p_tmp if dropout_mask is None else p_tmp.masked_fill(~dropout_mask, 0)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # qk_max1 = torch.max(qk[:, :, 128:, 192:], -1, keepdim=True).values
    # qk_max2 = torch.max(qk[:, :, 128:, 128:], -1, keepdim=True).values
    # qk_max3 = torch.max(qk[:, :, 128:, 64:], -1, keepdim=True).values
    # qk_max4 = torch.max(qk[:, :, 128:, :], -1, keepdim=True).values
    # o1 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 192:] - qk_max1) / math.sqrt(d)), v[:, 192:])
    # o2 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 128:] - qk_max2) / math.sqrt(d)), v[:, 128:])
    # o3 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 64:] - qk_max3) / math.sqrt(d)), v[:, 64:])
    # o4 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, :] - qk_max4) / math.sqrt(d)), v[:, :])
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    # do_o = (g.float() * out.float()).sum(-1)
    # dv_tmp = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, :64], g[:, :64])
    # dv_tmp1 = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, 64:], g[:, 64:])
    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        (dqkv,) = torch.autograd.grad(out, qkv, g)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        if not alibi:
            assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 257, 384, 512, 768, 1025, 2048])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_varlen_qkvpacked(
    seqlen, d, dropout_p, causal, local, alibi, deterministic, dtype
):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 6
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen, seqlen, key_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        *qkv.unbind(dim=2), key_padding_mask, key_padding_mask, qkvpacked=True
    )

    out_unpad, sm_lse, S_dmask = flash_attn_varlen_qkvpacked_func(
        qkv_unpad,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            key_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        attn = normalize_flash_attn_S(
            attn_unnorm,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            key_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask, key_padding_mask, key_padding_mask, causal=causal, window_size=window_size
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_qkvpacked_ref(
        qkv,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        (dqkv_unpad,) = torch.autograd.grad(out, qkv_unpad, g)
        dqkv = dqkv_pad_fn(dqkv_unpad)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        if not alibi:
            assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()


@pytest.mark.parametrize("kvpacked", [True, False])
# @pytest.mark.parametrize("kvpacked", [False])
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# @pytest.mark.parametrize("dropout_p", [0.17])
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        out, lse, S_dmask = flash_attn_kvpacked_func(
            q,
            kv,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        out, lse, S_dmask = flash_attn_func(
            q,
            k,
            v,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            None,
            None,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask, None, None, causal=causal, window_size=window_size
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        if kvpacked:
            (
                dq,
                dkv,
            ) = torch.autograd.grad(out, (q, kv), g)
            dk, dv = dkv.unbind(2)
            (
                dq_ref,
                dkv_ref,
            ) = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            (
                dq_pt,
                dkv_pt,
            ) = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            (
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(out_ref, (q, k, v), g)
            (
                dq_pt,
                dk_pt,
                dv_pt,
            ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        if not alibi:
            assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize("kvpacked", [True, False])
# @pytest.mark.parametrize('kvpacked', [False])
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize('mha_type', ["mqa"])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize("kv_cache_dtype", [None, torch.float8_e5m2])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked, kv_cache_dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if kvpacked and kv_cache_dtype:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        ) = generate_qkv(q, *kv.unbind(dim=2), query_padding_mask, key_padding_mask, kvpacked=True)
        out_unpad, sm_lse, S_dmask = flash_attn_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        out_unpad, sm_lse, S_dmask = flash_attn_varlen_func(
            q_unpad,
            maybe_fp8(k_unpad, kv_cache_dtype),
            maybe_fp8(v_unpad, kv_cache_dtype),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            maybe_fp8_ref(k, kv_cache_dtype, q.dtype),
            maybe_fp8_ref(v, kv_cache_dtype, q.dtype),
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_ref(
            q,
            maybe_fp8_ref(k, kv_cache_dtype, q.dtype),
            maybe_fp8_ref(v, kv_cache_dtype, q.dtype),
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    test_backward = False
    if test_backward and (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        if kvpacked:
            (
                dq_unpad,
                dkv_unpad,
            ) = torch.autograd.grad(out, (q_unpad, kv_unpad), g)
            dk, dv = dkv_pad_fn(dkv_unpad).unbind(2)
            (
                dq_ref,
                dkv_ref,
            ) = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            (
                dq_pt,
                dkv_pt,
            ) = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            (
                dq_unpad,
                dk_unpad,
                dv_unpad,
            ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(out_ref, (q, k, v), g)
            (
                dq_pt,
                dk_pt,
                dv_pt,
            ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dq = dq_pad_fn(dq_unpad)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        if not alibi:
            assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    if (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_causal(seqlen_q, seqlen_k, swap_sq_sk, d, local, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    causal = True
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = flash_attn_func(q, k, v, 0.0, causal=causal, window_size=window_size)
    out_ref, attn_ref = attention_ref(
        q, k, v, None, None, None, 0.0, None, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    test_backward = False
    if test_backward and (d <= MAX_HEADDIM_SM8x or d > 224) or (is_sm80 or is_sm90):
        (
            dq,
            dk,
            dv,
        ) = torch.autograd.grad(out, (q, k, v), g)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    if (d <= MAX_HEADDIM_SM8x or d > 224) or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 1e-5
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 1e-5
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# TODO: add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged
@pytest.mark.parametrize("paged_kv_block_size", [16, 256, 512])
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 128)])
@pytest.mark.parametrize("kv_cache_dtype", [torch.float8_e5m2])
def test_flash_attn_varlen_causal(
    seqlen_q, seqlen_k, swap_sq_sk, d, local, paged_kv_block_size, dtype, kv_cache_dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    causal = True
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)

    if paged_kv_block_size is None:
        k = torch.randn(
            batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        block_table = None
    else:
        k, v, block_table, k_cache_paged, v_cache_paged, num_blocks = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads, d, device, dtype
        )
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    out_unpad = flash_attn_varlen_func(
        q_unpad,
        maybe_fp8(k_unpad if paged_kv_block_size is None else k_cache_paged, kv_cache_dtype),
        maybe_fp8(v_unpad if paged_kv_block_size is None else v_cache_paged, kv_cache_dtype),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    out = output_pad_fn(out_unpad)
    out_ref, attn_ref = attention_ref(
        q,
        maybe_fp8_ref(k, kv_cache_dtype, q.dtype),
        maybe_fp8_ref(v, kv_cache_dtype, q.dtype),
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        maybe_fp8_ref(k, kv_cache_dtype, q.dtype),
        maybe_fp8_ref(v, kv_cache_dtype, q.dtype),
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    test_backward = False
    if test_backward:
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    if test_backward:
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 1e-5
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 1e-5
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (3, 1024),
        (1, 339),
        (64, 800),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (16, 100000),
        (128, 128),
        (256, 256),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", [None, torch.float8_e5m2])
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_splitkv(
    seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, alibi, deterministic, dtype, kv_cache_dtype
):
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 12
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    out, lse, _ = flash_attn_func(
        q,
        maybe_fp8_(k, kv_cache_dtype),
        maybe_fp8(v, kv_cache_dtype),
        0.0,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    out_ref, attn_ref = attention_ref(
        q, maybe_fp8_ref(k, kv_cache_dtype, q.dtype), maybe_fp8_ref(v, kv_cache_dtype, q.dtype), None, None, attn_bias, 0.0, None, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_ref(
        q,
        maybe_fp8_ref(k, kv_cache_dtype, q.dtype),
        maybe_fp8_ref(v, kv_cache_dtype, q.dtype),
        None,
        None,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    test_backward = False
    if test_backward and (d <= MAX_HEADDIM_SM8x or d > 224) or (is_sm80 or is_sm90):
        (
            dq,
            dk,
            dv,
        ) = torch.autograd.grad(out, (q, k, v), g)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    mult = 2 if not alibi else 8
    if (d <= MAX_HEADDIM_SM8x or d > 224) or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= mult * (dq_pt - dq_ref).abs().max().item() + 2e-4
        assert (dk - dk_ref).abs().max().item() <= mult * (dk_pt - dk_ref).abs().max().item() + 2e-4
        assert (dv - dv_ref).abs().max().item() <= mult * (dv_pt - dv_ref).abs().max().item() + 2e-4


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_splits", [0])
# @pytest.mark.parametrize("num_splits", [1])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("new_kv", [False])
# @pytest.mark.parametrize("new_kv", [False])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
@pytest.mark.parametrize("rotary_interleaved", [False, True])
# @pytest.mark.parametrize("rotary_interleaved", [False])
@pytest.mark.parametrize("rotary_fraction", [0.0])
# @pytest.mark.parametrize("rotary_fraction", [0.0])
# @pytest.mark.parametrize("paged_kv_block_size", [None, 256, 512])
@pytest.mark.parametrize("paged_kv_block_size", [16, 256, 512])
@pytest.mark.parametrize("has_batch_idx", [False])
# @pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("batch_size", [2, 4, 8, 32])
@pytest.mark.parametrize("kv_cache_dtype", [torch.float8_e5m2])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (1, 128 * 1024),
        (16, 128 * 1024),
        (128, 128),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    mha_type,
    num_splits,
    dtype,
    kv_cache_dtype,
    batch_size,
):
    if rotary_fraction != 0 and kv_cache_dtype:
        pytest.skip()
    if paged_kv_block_size is None and kv_cache_dtype:
        pytest.skip()
    if new_kv and kv_cache_dtype:
        pytest.skip()
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if has_batch_idx and paged_kv_block_size is not None:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
            :batch_size
        ]
    else:
        cache_batch_idx = None
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, None, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = (
            torch.rand(
                seqlen_k if paged_kv_block_size is None else num_blocks * paged_kv_block_size,
                rotary_dim // 2,
                device=device,
            )
            * 2
            * math.pi
        )
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (
        k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    v_cache_ref = (
        v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    out = flash_attn_with_kvcache(
        q,
        maybe_fp8(k_cache if paged_kv_block_size is None else k_cache_paged, kv_cache_dtype),
        maybe_fp8(v_cache if paged_kv_block_size is None else v_cache_paged, kv_cache_dtype),
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        block_table=block_table,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
    )
    # out = flash_attn_with_kvcache(
    #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
    # )
    # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
    # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # probs = torch.softmax(qk, dim=-1)
    out_ref, _ = attention_ref(
        q_ro,
        maybe_fp8_ref(k_cache_rep, kv_cache_dtype, q.dtype),
        maybe_fp8_ref(v_cache_rep, kv_cache_dtype, q.dtype),
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, _ = attention_ref(
        q_ro,
        maybe_fp8_ref(k_cache_rep, kv_cache_dtype, q.dtype),
        maybe_fp8_ref(v_cache_rep, kv_cache_dtype, q.dtype),
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = (
                k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
            )
            v_cache_select = (
                v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
            )
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    mult = 3 if not alibi else 5
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5


def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 56, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# @pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, dropout_p, causal, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    torch.random.manual_seed(42)
    out0, lse0, _ = flash_attn_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
    g = torch.randn_like(out0)
    test_backward = False
    if test_backward and (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
        (
            dq0,
            dk0,
            dv0,
        ) = torch.autograd.grad(out0, (q, k, v), g)
        # Numerical error if we just do any arithmetic on dq
        dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(250):
        torch.random.manual_seed(42)
        out, lse, _ = flash_attn_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
        assert torch.equal(out, out0)
        assert torch.equal(lse, lse0)

        if test_backward and (d <= MAX_HEADDIM_SM8x or (d > 224 and dropout_p == 0)) or (is_sm80 or is_sm90):
            (
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
            dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
            if not dq_equal:
                print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
            assert torch.equal(dv, dv0)
            assert torch.equal(dk, dk0)
            assert dq_equal


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [16, 32, 64])
# @pytest.mark.parametrize('d', [16])
@pytest.mark.parametrize("seqlen", [1, 2, 5, 17, 128])
# @pytest.mark.parametrize('seqlen', [2])
@pytest.mark.skip
def test_flash_attn_bwd_overflow(seqlen, d, causal, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
    in the case where seqlen % 128 != 0.
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 5
    q = torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 5
    k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 3
        for _ in range(2)
    ]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = flash_attn_func(q, k, v, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt, _ = attention_ref(q_pt, k_pt, v_pt, causal=causal, upcast=False, reorder_ops=True)
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, attn_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 5 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item() + 1e-3
    assert (k.grad - k_ref.grad).abs().max().item() <= 5 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item() + 1e-3
    assert (v.grad - v_ref.grad).abs().max().item() <= 5 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item() + 1e-3


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [64, 128])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 256])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.skip
def test_flash_attn_bwd_transpose(seqlen, d, causal, dtype):
    """We previously had a bug where we were using the wrong strides of dout, which shows up
    when dout is not contiguous.
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 2
    q, k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    out = rearrange(flash_attn_func(q, k, v, causal=causal), "b s ... -> s b ...")
    # So g is not contiguous
    g = torch.randn(seqlen, 2 * batch_size, nheads, d, dtype=dtype, device="cuda")[:, ::2]
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt, attn_pt = attention_ref(q_pt, k_pt, v_pt, causal=causal, upcast=False, reorder_ops=True)
    out_pt = rearrange(out_pt, "b s ... -> s b ...")
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, attn_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_ref = rearrange(out_ref, "b s ... -> s b ...")
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 2 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item()
    assert (k.grad - k_ref.grad).abs().max().item() <= 2 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item()
    assert (v.grad - v_ref.grad).abs().max().item() <= 2 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [16, 32, 64])
# @pytest.mark.parametrize('d', [16])
@pytest.mark.skip
def test_flash_attn_bwd_varlen_overflow(d, causal, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
    in the case where seqlen % 128 != 0 or varlen.
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    nheads = 5
    q_cuseqlen = torch.tensor([0, 76, 110, 256], device=device, dtype=torch.int32)
    k_cuseqlen = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    Mq = 256
    Mk = 3

    q = torch.randn([Mq, nheads, d], dtype=dtype, device=device) * 3
    k, v = [torch.randn([Mk, nheads, d], dtype=dtype, device=device) * 3 for _ in range(2)]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = flash_attn_varlen_func(q, k, v, q_cuseqlen, k_cuseqlen, Mq, Mk, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)

    assert not q.grad.isnan().any()
    assert not k.grad.isnan().any()
    assert not v.grad.isnan().any()


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = flash_attn_func(q, k, v, 0.0, causal=causal, window_size=window_size, deterministic=True)

    g = torch.randn_like(out)
    test_backward = False
    if test_backward and (d <= MAX_HEADDIM_SM8x or d > 224) or (is_sm80 or is_sm90):
        dq0, dk0, dv0 = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
        for _ in range(50):
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
            assert torch.equal(dv, dv0)
            assert torch.equal(dk, dk0)
            assert torch.equal(dq, dq0)


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", [None, torch.float8_e5m2])
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 128)])
def test_flash_attn_varlen_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype, kv_cache_dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    out = flash_attn_varlen_func(
        q_unpad,
        maybe_fp8(k_unpad, kv_cache_dtype),
        maybe_fp8(v_unpad, kv_cache_dtype),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
        window_size=window_size,
        deterministic=True,
    )

    g = torch.randn_like(out)
    test_backward = False
    if test_backward and (d <= MAX_HEADDIM_SM8x or d > 224) or (is_sm80 or is_sm90):
        dq, dk, dv = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
        for _ in range(50):
            dq, dk, dv = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
            assert torch.equal(dv, dv)
            assert torch.equal(dk, dk)
            assert torch.equal(dq, dq)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("paged_kv_block_size", [16])
# @pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("nheads", [32])
@pytest.mark.parametrize("b", [4])
@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(170, 170)])
@pytest.mark.parametrize("kv_cache_dtype", [None, torch.float8_e5m2])
def test_flash_attn_paged_kvcache_overflow(
    seqlen_q,
    seqlen_k,
    d,
    nheads,
    b,
    n,
    paged_kv_block_size,
    causal,
    dtype,
    kv_cache_dtype,
):  
    device = "cuda"
    num_blocks = 1000*16//paged_kv_block_size
    key_cache = torch.rand([num_blocks, paged_kv_block_size, nheads, d], dtype=dtype, device=device)
    value_cache = torch.rand([num_blocks, paged_kv_block_size, nheads, d], dtype=dtype, device=device)
    cache_seqlens = torch.zeros(b, dtype=torch.int32, device=device)

    for _ in range(n):
        query = torch.rand([b, seqlen_q, nheads, d], dtype=dtype, device=device)
        key = torch.rand([b, seqlen_k, nheads, d], dtype=dtype, device=device)
        value = torch.rand([b, seqlen_k, nheads, d], dtype=dtype, device=device)
        block_tables = torch.randint(0, num_blocks, size=(b, (seqlen_k + paged_kv_block_size - 1) // paged_kv_block_size), dtype=torch.int32, device=device)
        output = flash_attn_with_kvcache(
            query,
            maybe_fp8(key_cache, kv_cache_dtype),
            maybe_fp8(value_cache, kv_cache_dtype),
            k=key,
            v=value,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            causal=causal,
        )
