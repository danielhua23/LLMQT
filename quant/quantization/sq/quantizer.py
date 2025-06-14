import transformers
import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from quant.nn_models.modules.linear import (
    get_concrete_linear_module
)
from quant.utils.common_utils import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    clear_memory, 
    get_best_device
)
from quant.utils.sq_calib_utils import get_act_scales, get_static_decoder_layer_scales
from quant.quantization.base.quantizer import BaseQuantizer
from quant.quantization.sq.smooth import smooth_lm
# should check bias and output type fp32 or int32, w/ or w/o scaling
# sq里面的linear都带alpha和beta两个scale，只有W8A8B32O32LinearWithoutScaling不带
# W8A8B8O8LinearReLU用在mlp.fc1，输出的s8，喂到mlp.fc2继续做s8gemm
# W8A8BFP32OFP32Linear用在mlp.fc2和attn.out_proj
# W8A8B8O8Linear用在qkv，output为int8的原因是这个项目里面把qk_bmm设为了s8s8f32
# f32的attnscores经过softmax后再次quantize为了s8，pv_bmm设成了s8s8s8，pv的输出s8最后送到out proj输出fp32
# TODO 调研一下除了opt外的其他模型apply sq是如何做精度转换的？是否和opt一样？
# 考虑到int8sq涉及qdq以适应上下层的精度，所以多个cutlass s8 gemm是需要的
class SqQuantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # Qwen2ModelForCausal类 # only use for awq
        model,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data,
        duo_scaling,
        modules_to_not_convert=None,
        fake_quant=False, # true时为fake quant，false为real quant
        apply_clip=False,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.quant_method = quant_method
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.fake_quant = fake_quant
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        # self.modules, self.module_kwargs, self.inps = self.init_quant(
        #     n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        # )
    # fake quant tensor-wise
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self):
        # 对每个layer的input act求amax
        act_scales = get_act_scales(
            self.model, self.tokenizer, self.calib_data, self.max_calib_samples, self.max_calib_seq_len
        )
        # smooth
        smooth_lm(self.model, act_scales, 0.5)

        named_linears = get_named_linears(self.model)# self.modules[i])

        # Filter out the linear layers we don't want to exclude
        named_linears = exclude_layers_to_not_quantize(
            named_linears, self.modules_to_not_convert
        )
        # calib，与上面的get_act_scales的区别在于，这里还要求每个linear的output max，最后得到input scale和output scale，用以QDQ
        decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(self.model,
                                                                        self.tokenizer,
                                                                        self.calib_data,# args.dataset path得看看怎么换一下
                                                                        self.max_calib_samples,
                                                                        self.max_calib_seq_len)
        # [STEP 4]: scale和clip都apply之后，开始real Quantize weights+替换int8 linear
        if not self.fake_quant:
            self._apply_quant(self.model, decoder_layer_scales, named_linears) # TODO待调整

        clear_memory()

    def _apply_quant(self, module, decoder_layer_scales, named_linears: Dict[str, nn.Linear]):
        dev = get_best_device()
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(dev)#删除.half()

            # sq中不存在scales和zeros的transpose，只有awq中存在
            #     scales = scales.t().contiguous()
            #     if zeros is not None:
            #         zeros = zeros.t().contiguous()
            #     q_linear_module = WQLinear_GEMM

            q_linear_module = get_concrete_linear_module(self.quant_method) # WQLinear_GEMM
            # layer_id = int(name.split(".")[2]) # for qwen2 llamalike model
            proj = name.split(".")[-1]
            try:
                layer_id = int(name.split(".")[3])# for opt model
            except:
                layer_id = int(name.split(".")[2])# for non-opt model
            if proj in ["q_proj", "k_proj", "v_proj"]:
                print("[info] hit layer ", layer_id)
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["attn_input_scale"]
            
            # for qwen2 llamalike model
            elif proj in ["o_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["out_input_scale"]
            elif proj in ["gate_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["gate_input_scale"]
            elif proj in ["up_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["up_input_scale"]
            elif proj in ["down_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["down_input_scale"]
            
            # for opt
            elif proj in ["out_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["out_input_scale"]
            elif proj in ["fc1"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["fc1_input_scale"]
            elif proj in ["fc2"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["fc2_input_scale"]
            else:
                print("[warning] this dont hit any proj, pls check. the current is ", proj)

            q_linear = q_linear_module.from_linear(
                module=linear_layer,
                input_scale=scales, # input scale
                dev=dev,
                #zeros=zeros,
            )

            linear_layer.cpu()
            # q_linear.to(next(module.parameters()).device)
            q_linear.to(dev) # TODO 感觉这个to多余，weight等参数已经在init中to到了cuda
            set_op_by_name(module, name, q_linear)
            clear_memory()