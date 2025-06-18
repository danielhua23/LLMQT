import os
import torch
import logging
from transformers import AutoConfig
from quant.nn_models import *
from .base import BaseModelForCausalLM

Quant_CAUSAL_LM_MODEL_MAP = {
    "llama": LlamaModelForCausalLM, # TO BE SUPPORTED
    "opt": OptModelForCausalLM,
    "qwen2": Qwen2ModelForCausalLM,
    "qwen3": Qwen3ModelForCausalLM, # TO BE VALIDATED
    "qwen3_moe": Qwen3MoeModelForCausalLM, # TO BE VALIDATED
    "deepseek_v3": DeepseekV3ModelForCausalLM,
    "qwen2_distilled_r1": DeepseekR1DistillQwen2ModelForCausalLM,
    "llama4": Llama4MoeModelForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    config = AutoConfig.from_pretrained( # 返回config.json中的quantization_config字段
        model_dir, trust_remote_code=trust_remote_code, **model_init_kwargs
    )
    print("model type is ", config.model_type)
    if config.model_type not in Quant_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    if model_dir[:8] == "deepseek" and model_type == "qwen2":
        model_type = "qwen2_distilled_r1"
    return model_type


class AutoQuantForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "You must instantiate AutoQuantForCausalLM with\n"
            "AutoQuantForCausalLM.from_quantized or AutoQuantForCausalLM.from_pretrained, rather directly init this class"
        )

    @classmethod
    def from_pretrained(
        self,
        model_path, # 只有这一个入参，其他都取默认值
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        download_kwargs=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ) -> BaseModelForCausalLM:
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        return Quant_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            device_map=device_map,
            download_kwargs=download_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
            **model_init_kwargs,
        )