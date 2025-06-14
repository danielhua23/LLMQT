import transformers
import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict

from .scale import apply_scale, apply_clip

from quant.nn_models.modules.linear import (
    get_concrete_linear_module
)
import time
# sys.path.append("../../")
from quant.utils.calib_utils import get_calib_dataset
from quant.utils.common_utils import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    clear_memory, 
    get_best_device
)
# import sys
# sys.path.append("../")
from quant.quantization.base.quantizer import BaseQuantizer

# 整体步骤: init_quant拿到第0个decoderlayer的input act作为self.inps，
# 然后quantize对每个decoderlayer循环处理(calib，search best scale，apply scale，search best clip，apply clip，real quant)
class AwqQuantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # from pretained返回的model
        model,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data, # str
        duo_scaling, # true of false
        modules_to_not_convert=None,
        fake_quant=False, # true时为fake quant，false为real quant
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.awq_model = modelforCausalLM # Qwen2AwqForCausal类
        self.model = model
        self.tokenizer = tokenizer
        self.quant_method = quant_method
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.duo_scaling = duo_scaling
        self.fake_quant = fake_quant
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples + 128 # increase calib nums for qwen3 moe in case line 731 assert error
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        # import pdb;pdb.set_trace()
        # 返回值这里不能命名为self.modules，因为BaseQuantizer是一个torch.nn.Module，它也有self.modules成员
        self.target_modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )
    # fake quant
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape #[5120,5120]
        if self.group_size > 0: # for deepseek and group size > 0
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size) #[5120x40,128]
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        # 1.非对称量化的scale和zp的计算公式
        # scale = (absmax - absmin) / 255, zp = clip(-round(absmin/scale), 0, 255)
        # qx = clip(round(x / scale - zp))
        # 2.对称量化的scale和zp的计算公式
        # scale = absmax / 127, zp = 0
        # qx = clip(round(x / scale), -128, 127)        
        
        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True) # [5120x40,1]即列上每128个元素的最大值
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
        # import pdb;pdb.set_trace()
        scales = scales.view(org_w_shape[0], -1) # [5120x40,1]=>[5120,40]
        w = w.reshape(org_w_shape) #[5120x40,128]=>[5120,5120]

        return w, scales, zeros#[5120,40]

    # fake quant, 因为w的shape，我猜应该是[out feats, in feats]，所以其实这里的per group是针对每一列的
    def real_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape #[5120,5120]
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size) #[5120x40,128]
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True) # [5120x40,1]即列上每128个元素的最大值
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0
        # import pdb;pdb.set_trace()
        scales = scales.view(org_w_shape[0], -1) # [5120x40,1]=>[5120,40]
        w = w.reshape(org_w_shape) #[5120x40,128]=>[5120,5120]

        return w, scales, zeros#[5120,40]
    
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
        # self.modules, self.module_kwargs, self.inps = self.init_quant(
        #     n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        # )
        # 遍历每个decoderLayer
        
        for i in tqdm(range(len(self.target_modules)), desc="AWQ"): # init_quant返回了第0个decoder layers的input act
            start = time.perf_counter()
            # Move module and inputs to correct device（multi gpu）
            common_device = next(self.target_modules[i].parameters()).device # next的意思是获取该module的第一个参数
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count()) # 将当前第i个module的weight移动到第i个device
                else:
                    best_device = get_best_device()

                self.target_modules[i] = self.target_modules[i].to(best_device)
                common_device = next(self.target_modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)# init quant得到得第0个decoder layers的input act

            # We need to move the rotary embedding every time we move to a new module.
            # Transformers 4.45.0 moved rotary embedding to model definition as of this PR:
            # https://github.com/huggingface/transformers/pull/32617
            # dpsk解析：在较旧版本的 Transformers 中，rotary_embed（旋转位置编码）是 每个注意力层独立维护的，因此量化每个层时无需额外处理
            # 从 Transformers 4.45.0 开始（PR #32617），rotary_embed 被提升到 模型全局定义（以尽量达到全静态图适配torch compile），而不是每个层单独持有。
            # 目的：减少内存占用，避免重复计算。
            # 副作用：量化时需要显式确保 rotary_embed 和设备同步。
            # 如果4.45.0后, rotary_embed 是全局的，而某些层被移动到其他设备，会导致 设备不匹配错误，所以每个layer都需要显式移动rotary embed到指定设备
            self.awq_model.move_embed(self.model, common_device)
            # import pdb;pdb.set_trace()
            # 以下代码在deepseek v3的quantize中会crash，说找不到dpskv3找不到rotary_emb的属性，毕竟奇怪，估计是transformers的版本问题
            # Transformers >= 4.48.0 requires positional embeddings should be computed before forward pass 来自于https://github.com/casper-hansen/AutoAWQ/pull/706
            # if (
            #     transformers.__version__ >= "4.48.0"
            #     and self.module_kwargs.get("position_embeddings") is None
            # ):
            #     self.module_kwargs["position_embeddings"] = self.model.model.rotary_emb(
            #         self.inps, self.module_kwargs["position_ids"]
            #     )

            if (transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get('attention_mask') is None):
                self.module_kwargs['attention_mask'] = None

            for k, v in self.module_kwargs.items():
                # position embeddings found in tuple
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) 
                        else item for item in v
                    )

            # [STEP 1]: Get layer, extract linear target_modules, extract input features
            # {'self_attn.q_proj': Linear(in_features=5120, out_features=5120, bias=True), 
            # 'self_attn.k_proj': Linear(in_features=5120, out_features=1024, bias=True), 
            # 'self_attn.v_proj': Linear(in_features=5120, out_features=1024, bias=True), 
            # 'self_attn.o_proj': Linear(in_features=5120, out_features=5120, bias=False), 
            # 'mlp.gate_proj': Linear(in_features=5120, out_features=13824, bias=False), 
            # 'mlp.up_proj': Linear(in_features=5120, out_features=13824, bias=False), 
            # 'mlp.down_proj': Linear(in_features=13824, out_features=5120, bias=False)}
            named_linears = get_named_linears(self.target_modules[i])
            # import pdb;pdb.set_trace()
            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            # import pdb;pdb.set_trace()
            # calib，返回每个decoderlayer中每个linear的input features
            # dict_keys(['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'])
            # input_feat['self_attn.q_proj'].shape = [59,512,5120]
            input_feat = self._get_input_feat(self.target_modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            # (Pdb) module_config[0].keys()
            # dict_keys(['prev_op', 'layers', 'inp', 'module2inspect', 'kwargs'])
            # (Pdb) len(module_config)
            # 3 attn qkvo为1个，gate和up为1个，down为1个
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.target_modules[i], input_feat, self.module_kwargs
            )
            # (Pdb) scales_list[0]
            # ('input_layernorm' prev op name, ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj') layer name,
            #   best scales: tensor([1.3828, 1.2344, 1.2500,  ..., 1.2812, 1.1250, 1.1406], dtype=torch.bfloat16))
            scales_list = [ #搜寻每个linear的best scale
                self._search_best_scale(self.target_modules[i], **layer) # 数据都在module2inspect.parameter的device上面，即attn mlp这些module
                for layer in module_config
            ]
            # apply best scale, 这个scale是考虑到activation-aware后使得WX-Q(W)X误差最小的scale
            apply_scale(self.target_modules[i], scales_list, input_feat_dict=input_feat) # 这个使得所有weight都到了CPU
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.target_modules[i]) + "."
            )
            
            # 数据现在在cuda
            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip( # 数据在self.target_modules[i]中
                    self.target_modules[i], named_linears, input_feat
                )
                apply_clip(self.target_modules[i], clip_list) # 数据在self.target_modules[i]中
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.target_modules[i]) + "."
                )
            # import pdb;pdb.set_trace()
            # 数据现在在cuda
            # [STEP 4]: scale和clip都apply之后，开始real Quantize weights
            if not self.fake_quant:
                self._apply_quant(self.target_modules[i], named_linears)
            
            clear_memory()
            end = time.perf_counter()
            # 修改前249.35s
            # 修改后98.46s
            print("[info] the quantization time per layer is ", end - start)
    def pack(self):
        for i in tqdm(range(len(self.target_modules)), desc="Packing"):
            named_linears = get_named_linears(self.target_modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.target_modules_to_not_convert
            )
            self._apply_quant(self.target_modules[i], named_linears)
            clear_memory()

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            print("[info] hit ", name)
            
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.half()
            
            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )
            # 易错，上面的pseudo quantize只是为了拿出scales，real quantize会在linear awq.from linear做掉
            # !!!不过上面这里的weight我认为没有必要送进去fake quant一下，只需拿到scale和zero不就好了，后面可以试一下
            
            # 重要！！需要经过transpose
            scales = scales.t().contiguous()
            if zeros is not None:
                zeros = zeros.t().contiguous()

            q_linear_module = get_concrete_linear_module(self.quant_method) # WQLinear_GEMM
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()
    # 对每个layer作calib
    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad() # 数据都在module2inspect的device上面
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            # 下面是enable deepseek v3加上的
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale( # x mean和w mean是每个channel的mean
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    # 数据都在x的device上面
    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        # 求出了channel level的x mean和w mena后
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)
        # x mean和w mean是每个channel的mean
        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            # x mean越大，那么该channel的scale就越大，根据paper，量化误差就越小，因此不用mixed prec，还是全部量化为int4
            # 但是为什么最终scale是下式，这个还得查一查
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)：乘上scale又除scale，即fakequant，包含了量化误差
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X，module2inspect的意思是量化后需要fwd的module，以此来观察量化后对该module输出的影响
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            # 下面是enable deepseek v3加上的
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue
            # named_linears[name].to(get_best_device()) # 本身就在device上面
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model) # from pretained返回的model
        samples = get_calib_dataset(
            data=self.calib_data,
            # data="wikitext-2-v1",
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            # split=self.split,
            split="validation",
        )
        # import pdb;pdb.set_trace()
        samples = torch.cat(samples, dim=0)

        inps = []# list,里面只有一个元素，shape为[59,512,5120]
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)
        # 捕获到第0个module（第0个decoder layers）的input act到inps作为全局inps用到quantize
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try: # calibration以catch第0个module的输入,存到inps和layer_kwargs
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore
        # import pdb;pdb.set_trace()
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs) # 这个应该是hf里面modeling_qwen3.py里的方法
        # layer_kwargs: dict_keys(['cache_position', 'input_ids', 'inputs_embeds', 'position_ids', 'past_key_value', 'output_attentions', 'use_cache', 'position_embeddings'])
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]
        # import pdb;pdb.set_trace()
        modules[0] = modules[0].cpu() # modules为list，里面是每个decoderlayer module
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )
        # v2.8不存在下面代码
        # elif "qwen" in self.awq_model.model_type:
        #     layer_kwargs["attention_mask"] = None
        # import pdb;pdb.set_trace()
        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        # 这里是qwen3的改动，加了这行才能跑，猜测gate up的输入为named_linears[mlp]的值,后面打印一下layer看看
        if self.awq_model.model_type == "qwen3_moe" or self.awq_model.model_type == "deepseek_v3":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }
            
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        def cat_and_assert(k, v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"{k} has a zero dimension. This can happen if no data was passed through (e.g. an expert in MoE not being activated). "
                "Try increasing max_calib_samples (warning: this can significantly increase quantization time and memory usage.)"
            )
            return x

        input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
 过滤掉目标模块（module）的 forward 方法不支持的参数，确保传入的关键字参数（inputs_kwargs）
 不会因为transformers版本差异或参数不匹配导致模块的前向传播（forward）失败

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
