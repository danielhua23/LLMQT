import transformers
import torch
import copy
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from quant.nn_models.modules.linear import (
    get_concrete_linear_module,
    FP8StaticLinearQuantizer
)
from quant.nn_models.modules.linear.linear_fp8 import (
    per_tensor_quantize,
    static_per_tensor_quantize,
    replace_module
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
from quant.utils.fp8_calib_utils import *
from quant.quantization.base.quantizer import BaseQuantizer
# fp8不需要sq那么麻烦需要qdq搞来搞去，因为我们可以让fp8 gemm的output直接为fp16，至于accum需要为fp32的场景，这个后面看看cutlass的fp8 kernel应该是f8f8f32accumf16output
# 后续考虑per token activation + per channel weight， kernel对应就是scaled_mm，即rowwise gemm
class Fp8Quantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # only use for awq
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
        self.quant_method = quant_method
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.device = get_best_device() 
        # self.device = "cpu" #cpu上calib太慢
        self.dynamic_quant_linear = get_concrete_linear_module("fp8_dynamic_quant") # "fp8_dynamic_quant"
        
    # fake quant tensor-wise
    # def pseudo_quantize_tensor(self, w: torch.Tensor):
    #     org_w_shape = w.shape
    #     if self.group_size > 0:
    #         assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
    #         w = w.reshape(-1, self.group_size)
    #     assert w.dim() == 2
    #     assert torch.isnan(w).sum() == 0

    #     # zero point quantization
    #     if self.zero_point:
    #         max_val = w.amax(dim=1, keepdim=True)
    #         min_val = w.amin(dim=1, keepdim=True)
    #         max_int = 2**self.w_bit - 1
    #         min_int = 0
    #         scales = (max_val - min_val).clamp(min=1e-5) / max_int
    #         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    #         w = (
    #             torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    #         ) * scales
    #         zeros = zeros.view(org_w_shape[0], -1)
    #     else:
    #         max_val = w.abs().amax(dim=1, keepdim=True)
    #         max_val = max_val.clamp(min=1e-5)
    #         max_int = 2 ** (self.w_bit - 1) - 1
    #         min_int = -(2 ** (self.w_bit - 1))
    #         scales = max_val / max_int
    #         zeros = None
    #         w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    #     assert torch.isnan(scales).sum() == 0
    #     assert torch.isnan(w).sum() == 0

    #     scales = scales.view(org_w_shape[0], -1)
    #     w = w.reshape(org_w_shape)

    #     return w, scales, zeros

    # def pseudo_dequantize_tensor(
    #     self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    # ):
    #     # get repeated count
    #     repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
    #     scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

    #     # dequantize
    #     if self.zero_point:
    #         zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
    #         w = (w.weight.data - zeros) * scales
    #     else:
    #         w = w.weight.data * scales

    #     return w

    def quantize(self):
        named_modules = list(self.model.named_modules()) # self.model为AutoModelFromCausal.from_pretained的返回值
        calib_tokens = prepare_calib_tokens(self.tokenizer, self.device, self.max_calib_samples, self.max_calib_seq_len) # tokenizer是AutoTokenizer.from_pretrained(model path)的返回值
        for name, linear in tqdm(named_modules, desc="FP8 Quantizing weights"):
            if (
                not isinstance(linear, torch.nn.Linear)
                or name in self.quant_config.modules_to_not_convert
            ):
                print("=== skipping ", name)
                continue
            print("=== Dynamic Quantizing ", name)
            # quant_weight.shape, linear.weight.shape= [5120, 5120], weight_scale=tenosr(0.008)
            quant_weight, weight_scales = per_tensor_quantize(linear.weight) # 这一步多余
            bias = copy.deepcopy(linear.bias) if linear.bias is not None else None
            # TODO:把per tensor加入quant config
            q_linear = self.dynamic_quant_linear.from_linear(linear, weight=quant_weight, weight_scales=weight_scales, bias=bias, per_tensor=self.quant_config.per_tensor)
            # quant_linear = FP8DynamicLinear(
            #     weight=quant_weight, weight_scale=weight_scale, bias=bias
            # )
            replace_module(self.model, name, q_linear)
            del linear.weight
            del linear.bias
            del linear

        # [STEP 4]: scale和clip都apply之后，开始real Quantize weights+替换int8 linear
        if self.quant_config.per_tensor and (self.quant_method == "fp8_static_quant" or self.quant_config.fp8_static_quant):
            self._apply_quant_act(self.quant_config, calib_tokens) # TODO待调整
        else:
            print("[info] skip static quant, since per_tensor=False or quant method is not static quant")
        clear_memory()

    def _apply_quant_act(self, quant_config, calib_tokens):
        # Replace weight quantizer with a dynamic activation quantizer observer
        for name, dynamic_quant_linear in self.model.named_modules():
            if (
                not isinstance(dynamic_quant_linear, self.dynamic_quant_linear)
                or name in quant_config.modules_to_not_convert
            ):
                continue
            quantizer = FP8StaticLinearQuantizer(
                in_features=dynamic_quant_linear.in_features,
                out_features=dynamic_quant_linear.out_features,
                qdtype=dynamic_quant_linear.qdtype,
                weight=dynamic_quant_linear.weight,
                weight_scale=dynamic_quant_linear.weight_scale,
                bias=dynamic_quant_linear.bias,
                quantize_output=(
                    hasattr(quant_config, "kv_cache_quant_layers")
                    and name in quant_config.kv_cache_quant_layers
                ),
            )
            replace_module(self.model, name, quantizer)
            del dynamic_quant_linear
        clear_memory()
        # calibration
        # Pass through calibration data to measure activation scales
        self.model.to(self.device)
        with torch.inference_mode():
            with tqdm(total=calib_tokens.shape[0], desc="Calibrating activation scales") as pbar:
                for row_idx in range(calib_tokens.shape[0]):
                    self.model(calib_tokens[row_idx].reshape(1, -1))
                    clear_memory()
                    pbar.update(1)
        static_quant_linear = get_concrete_linear_module("fp8_static_quant")
        # Replace dynamic quantizer observer with StaticLinear for export
        for name, quantizer in self.model.named_modules():
            if (
                not isinstance(quantizer, FP8StaticLinearQuantizer)
                or name in quant_config.modules_to_not_convert
            ):
                print("=== skipping ", name)
                continue
            print("=== static Quantizing ", name)
            static_proj = static_quant_linear.from_linear(
                in_features=quantizer.in_features,
                out_features=quantizer.out_features,
                fp8_weight=quantizer.qweight,
                weight_scales=quantizer.weight_scale,
                bias=quantizer.bias,
                input_scale=quantizer.input_scale,
                output_scale=quantizer.output_scale,
                quantize_output=(
                    hasattr(quant_config, "kv_cache_quant_layers")
                    and name in quant_config.kv_cache_quant_layers
                ),
            )
            replace_module(self.model, name, static_proj)
            del quantizer
        clear_memory()

        # Post-process step for kv cache scales to take the k/v module
        # `output_scale` parameters, and store them in the parent attention
        # module as `k_scale` and `v_scale`
        if quant_config.kv_cache_quant_layers:
            # Assumes that list is ordered such that [layer0.k_proj, layer0.v_proj, layer1.k_proj, layer1.v_proj, ...]
            # so we make a list of tuples [(layer0.k_proj, layer0.v_proj), (layer1.k_proj, layer1.v_proj), ...]
            kv_proj_pairs = zip(*[iter(quant_config.kv_cache_quant_layers)]*2)
            for k_proj_name, v_proj_name in kv_proj_pairs:
                parent_module_name = ".".join(k_proj_name.split(".")[:-1])
                assert parent_module_name == ".".join(v_proj_name.split(".")[:-1])
                parent_module = dict(model.named_modules())[parent_module_name]

                k_proj = dict(model.named_modules())[k_proj_name]
                v_proj = dict(model.named_modules())[v_proj_name]
                # ！！！核心：量化kv在于把kv cache scale保存到k proj和v proj的parent module的属性中
                parent_module.k_scale = torch.nn.Parameter(k_proj.output_scale, requires_grad=False)
                parent_module.v_scale = torch.nn.Parameter(v_proj.output_scale, requires_grad=False)

                # Remove output_scale from k_proj and v_proj
                k_proj.output_scale = None
                v_proj.output_scale = None
        clear_memory()
            