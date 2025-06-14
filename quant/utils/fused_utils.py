import torch

from runtime.autoAWQ_models.modules.linear import (
    SqW8A8BBF16OBF16Linear,
    # WQLinear_GEMV,
    # WQLinear_GEMVFast,
)
# !!!在quant中，这个文件不会被使用
# 更新cache start pos
def prepare_cache(blocks, seqlen: int) -> int:
    for block in blocks:
        start_pos = block.attn.start_pos
        will_cache_be_exceeded = start_pos + seqlen > block.attn.max_seq_len

        # Reset and avoid retaining state when processing context
        if seqlen > 1 and (will_cache_be_exceeded or start_pos > 0):
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(
                start_pos, n=start_pos
            )

        # Slowly roll out old tokens without performance hit if exceeded during decoding
        elif seqlen == 1 and will_cache_be_exceeded:
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(start_pos, n=100)


def prepare_input_ids(input_ids: torch.Tensor, last_forward_num_tokens: int):
    # NOTE: from transformers 4.35.0, input_ids includes full context during decoding
    num_input_tokens = input_ids.shape[-1]
    num_new_tokens = num_input_tokens

    if num_input_tokens != 1:
        num_new_tokens = num_input_tokens - last_forward_num_tokens

        # after context is processed, slice to latest token
        if num_new_tokens == 1:
            input_ids = input_ids[:, -1:]

    return input_ids, last_forward_num_tokens + num_new_tokens

# 完成qkv linear的fuse以及替换为quantlinear
def fuse_qkv(module, q_proj, k_proj, v_proj):
    bias = (
        torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
        if q_proj.bias is not None
        else None
    )

    if isinstance(q_proj, SqW8A8BBF16OBF16Linear):
        q_linear = SqW8A8BBF16OBF16Linear
    else:
        print("[error!!] the linear module of the quantized model is not SqW8A8BBF16OBF16Linear")
    # elif isinstance(q_proj, WQLinear_GEMVFast):
    #     q_linear = WQLinear_GEMVFast

    qkv_layer = q_linear(
        q_proj.w_bit,
        q_proj.group_size,
        q_proj.in_features,
        q_proj.out_features + k_proj.out_features + v_proj.out_features,
        q_proj.bias is not None,
        next(iter(module.state_dict().values())).device,
    )

    # if isinstance(q_proj, WQLinear_GEMV):
    #     qkv_layer.qweight = torch.cat(
    #         [q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=0
    #     )
    #     qkv_layer.qzeros = torch.cat(
    #         [q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=0
    #     )
    #     qkv_layer.scales = torch.cat(
    #         [q_proj.scales, k_proj.scales, v_proj.scales], dim=0
    #     )
    #     qkv_layer.split_k_iters = q_proj.split_k_iters
    if isinstance(q_proj, SqW8A8BBF16OBF16Linear):
        qkv_layer.qweight = torch.cat(
            [q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1
        )
        qkv_layer.qzeros = torch.cat(
            [q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1
        )
        qkv_layer.scales = torch.cat(
            [q_proj.scales, k_proj.scales, v_proj.scales], dim=1
        )
    else:
        print("[error!!] the linear module of the quantized model is not SqW8A8BBF16OBF16Linear")
    qkv_layer.bias = bias

    for layer in [q_proj, k_proj, v_proj]:
        del (layer.qweight, layer.qzeros, layer.scales)

    return qkv_layer


def fuse_linears(linears, device, dim=1, operation=torch.cat):
    total_out_features = sum([layer.out_features for layer in linears])
    fused = SqW8A8BBF16OBF16Linear(
        linears[0].w_bit,
        linears[0].group_size,
        linears[0].in_features,
        total_out_features,
        bias=None,
        dev=device,
    )
    fused.qweight = operation([layer.qweight for layer in linears], dim=dim)
    fused.qzeros = operation([layer.qzeros for layer in linears], dim=dim)
    fused.scales = operation([layer.scales for layer in linears], dim=dim)

    for layer in linears:
        del (layer.qweight, layer.qzeros, layer.scales, layer)

    return fused


def get_attention_shapes(
    attention_shapes, n_heads, n_kv_heads, head_dim
):
    if attention_shapes is not None:
        attention_shapes = attention_shapes

    elif n_kv_heads == 0:
        attention_shapes = {
            "xqkv_view": (-1, n_heads, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0],
            "xk_slice": lambda xqkv: xqkv[:, :, 1],
            "xv_slice": lambda xqkv: xqkv[:, :, 2],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_heads, head_dim),
            "xv_view": (n_heads, head_dim),
            "xk_reshape": (n_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_heads, head_dim),
            "single_xv_view": (n_heads, head_dim),
        }

    else:
        attention_shapes = {
            "xqkv_view": (n_heads + n_kv_heads * 2, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0:n_heads],
            "xk_slice": lambda xqkv: xqkv[:, :, n_heads : (n_heads + n_kv_heads)],
            "xv_slice": lambda xqkv: xqkv[:, :, -n_kv_heads:],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_kv_heads, head_dim),
            "xv_view": (n_kv_heads, head_dim),
            "xk_reshape": (n_kv_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_kv_heads, head_dim),
            "single_xv_view": (n_kv_heads, head_dim),
        }

    return attention_shapes
