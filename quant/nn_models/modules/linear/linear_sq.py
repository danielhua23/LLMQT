# 仿照linear_Awq
import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from quant.utils.common_utils import get_best_device
from quant.utils.packing_utils import dequantize_gemm
from quant.utils.quantization_utils import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)

# should check bias and output type fp32 or int32, w/ or w/o scaling
# sq里面的linear都带alpha和beta两个scale，只有W8A8B32O32LinearWithoutScaling不带
# W8A8B8O8LinearReLU用在mlp.fc1，输出的s8，喂到mlp.fc2继续做s8gemm
# W8A8BFP32OFP32Linear用在mlp.fc2和attn.out_proj
# W8A8B8O8Linear用在qkv，output为int8的原因是这个项目里面把qk_bmm设为了s8s8f32
# f32的attnscores经过softmax后再次quantize为了s8，pv_bmm设成了s8s8s8，pv的输出s8最后送到out proj输出fp32
# TODO 调研一下除了opt外的其他模型apply sq是如何做精度转换的？是否和opt一样？

# sq官方试验了llama3，mixtral，mistral，那我们这里就拿llama3/qwen3来实验
# https://github.com/mit-han-lab/smoothquant/blob/main/examples/smoothquant_llama_demo.ipynb
# https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py
# torch.matmul默认只支持s8s8 in s32 out
# 对于s8s8 in f32/s8 out，需要手动将s32 requantize/dequantize为s8/fp32，如下

# from torch.ao.nn.quantized import Quantize, DeQuantize
# # 定义量化/反量化层
# quantize = Quantize(scale=0.1, zero_point=0, dtype=torch.qint8)
# dequantize = DeQuantize()
# # 假设已有 int8 输入和权重
# input_int8 = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
# weight_int8 = torch.randint(-128, 127, (4, 3), dtype=torch.int8)
# # 手动模拟线性层计算
# output_int32 = torch.matmul(input_int8, weight_int8.t())  # 转为 float 避免溢出
# output_fp32 = dequantize(output_int32)  # 反量化（此处仅为示例，需结合真实量化参数）

user_has_been_warned = False
from .linear_base import LinearBase
class SqW8A8BBF16OBF16Linear(LinearBase):
    # For qkv_proj
    def __init__(self, in_features, out_features, bias, weight_scale=1.0, input_scale=1.0, alpha=1.0, beta=1.0, dev="cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 仅trans，不改变majorness方向
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False,
                                                                device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (self.out_features), dtype=torch.float16, requires_grad=False,device=dev)) # qwen2是bf16,opt是fp16
        else:
            self.bias = None
        self.register_buffer('weight_scale', torch.tensor(weight_scale,device=dev)) 
        self.register_buffer('input_scale', torch.tensor(input_scale,device=dev))

    # TODO 感觉这个to多余，weight等参数已经在init中to到了cuda
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.qweight = self.qweight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)
        x_bf16 = x.to(torch.bfloat16) * self.input_scale.item()  # 反量化输入
        weight_bf16 = self.qweight.to(torch.bfloat16) * self.weight_scale.item()  # 反量化权重
        weight_bf16 = weight_bf16.t()
        y = torch.matmul(x_bf16, weight_bf16)  # FP16 计算
        if self.bias:
            y += self.bias #[xx, out feats] + [1, out feats]
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_linear(module: torch.nn.Linear, input_scale, dev="cuda:0"):
        # TODO: 添加一个入参,判断走perchannel还是pertensor,然后定义对应的weight scale和input scale shape
        int8_module = SqW8A8BBF16OBF16Linear(
            module.in_features, module.out_features, module.bias is not None)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        
        # 这里无论选择PerTensor或者PerChannel，weight scale的shape都要和register buffer的weight scale对的上才行
        # int8_weight, weight_scale = quantize_weight_per_channel_absmax(module.weight)
        int8_module.weight_scale = torch.tensor(weight_scale, device=dev) # scalar
        int8_module.input_scale = torch.tensor(input_scale, device=dev) # scalar
        int8_module.qweight = int8_weight # [out, in] row major
        if module.bias is not None:
            int8_module.bias = module.bias.clone()
        return int8_module