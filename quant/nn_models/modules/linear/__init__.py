from .linear_base import LinearBase
from .linear_awq import WQLinear_GEMM
from .linear_sq import SqW8A8BBF16OBF16Linear
from .linear_fp8 import FP8DynamicLinear, FP8StaticLinear, FP8StaticLinearQuantizer
method_to_linear: dict[str, type[LinearBase]] = {
    "awq": WQLinear_GEMM,
    "sq": SqW8A8BBF16OBF16Linear,
    "fp8_static_quant": FP8StaticLinear, # per tensor
    "fp8_dynamic_quant": FP8DynamicLinear, # per tensor
    #TODO: support more grained quantization
}
def get_concrete_linear_module(quant_method):
    return method_to_linear[quant_method]