import os
import gc
import warnings
import torch
import transformers
import torch.nn as nn

from tqdm import tqdm
from typing import List, Union, Dict
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download, save_torch_state_dict

from quant.nn_models.modules.linear import ( 
    WQLinear_GEMM, # csrc triton/cuda kernels
    get_concrete_linear_module,
)

from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoProcessor,
    BaseImageProcessor,
    ProcessorMixin,
    PreTrainedTokenizer,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from .config import QuantConfig # both need in quantizer and runtime
from quant.quantization import get_concrete_quantizer_cls

TRANSFORMERS_AUTO_MAPPING_DICT = {
    "llama": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "qwen2": "AutoModelForCausalLM",
    "qwen2_vl": "AutoModelForVision2Seq",
    "qwen3": "AutoModelForCausalLM",
    "qwen3_moe": "AutoModelForCausalLM",
}


class BaseModelForCausalLM(nn.Module):
    def __init__(
        self,
        model: Annotated[PreTrainedModel, Doc("The pretrained or quantized model.")],
        model_type: Annotated[str, Doc("The model type, found in config.json.")],
        is_quantized: Annotated[
            bool, Doc("Indicates if the current model is quantized.")
        ],
        config: Annotated[PretrainedConfig, Doc("The config of the model.")],
        quant_config: Annotated[
            QuantConfig, Doc("The quantization config of the model.")
        ],
        processor: Annotated[
            BaseImageProcessor, Doc("An optional processor, e.g. for vision models.")
        ],
    ):
        """The base model for all models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: QuantConfig = quant_config
        self.processor: ProcessorMixin = processor

    def to(self, device: Annotated[str, Doc("The device to move your model to.")]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        torch_dtype: Annotated[
            torch.dtype,
            Doc(
                "The dtype to load the model as. May not work with other values than float16."
            ),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc(
                "Useful for Huggingface repositories that have not been integrated into transformers yet."
            ),
        ] = True,
        safetensors: Annotated[
            bool, Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "A device map that will be passed onto the model loading method from transformers."
            ),
        ] = "auto",
        download_kwargs: Annotated[
            Dict,
            Doc("Used for configure download model"),
        ] = None,
        low_cpu_mem_usage: Annotated[
            bool,
            Doc("Use low_cpu_mem_usage when loading from transformers.")
        ] = True,
        use_cache: Annotated[
            bool,
            Doc("Use use_cache argument in transformers")
        ] = False,
        **model_init_kwargs: Annotated[
            Dict,
            Doc(
                "Additional kwargs that are passed to the model during initialization."
            ),
        ],
    ):
        """A method for initialization of pretrained models, usually in FP16."""
        # Get weights path and quant config by AutoConfig and QuantConfig
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            "",
            safetensors,
            trust_remote_code=trust_remote_code,
            download_kwargs=download_kwargs,
        )
        
        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)
        print("target cls name is ", target_cls_name)
        processor = None
        # if target_cls_name == "AutoModelForVision2Seq" or target_cls_name == "AutoModelForTextToWaveform":
        #     processor = AutoProcessor.from_pretrained(model_weights_path)
        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if model_init_kwargs.get("use_cache") is None and not ((target_cls_name == "AutoModelForVision2Seq") or (target_cls_name == "AutoModelForTextToWaveform")):
            model_init_kwargs["use_cache"] = use_cache

        # If not quantized, must load with AutoModelForCausalLM
        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,
        )

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=False,
            config=config,
            quant_config=quant_config,
            processor=processor,
        )

    # 其实就是real quant
    @torch.no_grad()
    def pack(self):
        """
        A utility function for the following scenario. Note that save_quantized will
        overwrite existing weights if you use the same quant_path.
        其实就是real quant，如果你想既保存fake quant，也想保存real quant，那么就像下列那样
        Example:

        ```python
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            fake_quant=True
        )
        model.save_quantized(...)  # produces GGUF/other compat weights
        model.pack(...) # makes the model CUDA compat
        model.save_quantized(...)  # produces CUDA compat weights
        ```
        """
        self.quantizer.pack()
        
    @torch.no_grad()
    def quantize( # 只传入tokenizer和quant config，其他使用默认值
        self,
        tokenizer: Annotated[
            PreTrainedTokenizer, Doc("The tokenizer to use for quantization.")
        ] = None,
        quant_config: Annotated[
            Dict, Doc("The quantization config you want to use.")
        ] = {},
        calib_data: Annotated[
            Union[str, List[str]],
            Doc(
                "The calibration dataset. Either a string pointing to Huggingface or a list of preloaded examples."
            ),
        ] = "pileval",
        duo_scaling: Annotated[
            bool, Doc("Whether to scale using both w/x or just x.")
        ] = True,
        fake_quant: Annotated[
            bool,
            Doc(
                "This argument avoids real quantization by only applying the scales without quantizing down to FP16."
            ),
        ] = False,
        apply_clip: Annotated[
            bool,
            Doc(
                "Whether to apply clipping to the model during quantization. Some models may perform better with this set to False."
            ),
        ] = True,
        n_parallel_calib_samples: Annotated[
            int,
            Doc(
                "The number of parallel samples to run through the model. "
                "A high number of parallel samples can result in OOM during quantization if max_calib_samples is high enough. "
                "If None, runs through all samples at the same time. "
                "You can set this to a low number for more memory efficient quantization."
            ),
        ] = None,
        max_calib_samples: Annotated[
            int, Doc("The maximum number of samples to run through the model.")
        ] = 128,
        max_calib_seq_len: Annotated[
            int,
            Doc(
                "The maximum sequence length of the calibration dataset. Discard samples greater than max_calib_seq_len."
            ),
        ] = 512,
        max_chunk_memory: Annotated[
            int,
            Doc(
                "The loss computation and per-channel mean is optimized into chunked computations."
                " Adjust this parameter to increase or decrease memory usage for these computations."
                " Default is 1GB (1024 * 1024 * 1024)."
            ),
        ] = 1024
        * 1024
        * 1024,
        **kwargs,
    ):
        """
        The main quantization function that you can use to quantize your model.
        """
        self.quant_config: QuantConfig = QuantConfig.from_dict(quant_config)
        
        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert
        quantizer_cls = get_concrete_quantizer_cls(self.quant_config.quant_method)
        self.quantizer = quantizer_cls(
            self, # models/下面的Qwen2ModelForCausal, only use for awq
            self.model,
            tokenizer,
            self.quant_config,
            self.quant_config.quant_method,
            self.quant_config.w_bit,
            self.quant_config.q_group_size, # only for awq
            self.quant_config.zero_point, # only for awq, sq is for homework
            calib_data, # use for sq and awq, dataset name
            duo_scaling, # only for awq
            modules_to_not_convert=self.quant_config.modules_to_not_convert,
            fake_quant=fake_quant, # use for awq and sq, but can change name to fake_quant
            apply_clip=apply_clip, # only for awq
            n_parallel_calib_samples=n_parallel_calib_samples, # only for awq
            max_calib_samples=max_calib_samples,
            max_calib_seq_len=max_calib_seq_len,
            max_chunk_memory=max_chunk_memory, # only for awq
            **kwargs,
        )
        # import pdb;pdb.set_trace()
        self.quantizer.quantize()

        self.is_quantized = True
        
    def save_quantized(
        self,
        save_dir: Annotated[str, Doc("The directory to save your model to.")],
        safetensors: Annotated[
            bool, Doc("Whether to save the model as safetensors or torch files.")
        ] = True,
        shard_size: Annotated[
            str, Doc("The shard size for sharding large models into multiple chunks.")
        ] = "5GB",
    ):
        save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model and config files with empty state dict
        self.model.config.quantization_config = self.quant_config.to_transformers_dict()
        self.model.generation_config.do_sample = True
        self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

        # Vision transformers have a processor
        if self.processor is not None:
            self.processor.save_pretrained(save_dir)

        # Remove empty state dict
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict(
            state_dict=self.model.state_dict(),
            save_directory=save_dir,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=self.model._tied_weights_keys,
        )

    def _load_config(
        self,
        model_path,
        model_filename,
        safetensors=True,
        trust_remote_code=True,
        max_seq_len=4096,
        download_kwargs=None,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            if download_kwargs is None:
                download_kwargs = {}

            if "ignore_patterns" in download_kwargs:
                download_kwargs_ignore_patterns = download_kwargs.pop("ignore_patterns")

                if isinstance(download_kwargs_ignore_patterns, str):
                    ignore_patterns.append(download_kwargs_ignore_patterns)
                elif isinstance(download_kwargs_ignore_patterns, list):
                    ignore_patterns.extend(download_kwargs_ignore_patterns)
            # 下载模型到/root/.cache/huggingface/hub
            model_path = snapshot_download(
                model_path, ignore_patterns=ignore_patterns, **download_kwargs
            )

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else:
            model_weights_path = model_path

        # [STEP 2] Load config and set sequence length
        # TODO: Create BaseAWQConfig class
        quant_config = QuantConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            # To add the generate support for Multi-modal models as well
            if hasattr(config, "text_config"):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048
                )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len

        return model_weights_path, config, quant_config