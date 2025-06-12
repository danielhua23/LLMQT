* awq: 
    * 支持per row quant，或者row上的per group quant，当group wise为0时为per row quant
    * 支持zero point即asym量化
    * 支持8xint4 weight pack为int32 weight
    * kernel为naive kernel，python dq + torch matmul
    * bug: nn.Module也有self.modules属性，所以AwqQuantizer的init_quant返回的第一个参数不能是self.modules,quantizer.py/L82
    * quant没有问题
* fp8:
    * dynamic quant: per tensor
    * static quant: per tensor
    * kv cache scale
    * scaled_mm不确定是否支持block scale gemm，目前row wise scale和tensor wise scale gemm肯定是可以的
    * fp8 naive gemm已考虑进了bias，不关心bias是什么dtype，直接拿linear.bias出来丢进去
    
* sq
    * bias问题，bf16或者fp16要看具体的模型weight dtype, qwen2为bf16
    * smooth.py中把act需要除以的scale融合进了layernorm的weight，这样后面真正进来的act就不用再除这个scale
    * tensor wise smooth
    * tensor wise static/dynamic quant支持
    * channel wise weight quant支持
    * static channel wise activation不可行，calibration的时候每个样本长度不一致，导致无法得到一个统一的channel wise scale，另一方面根据https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md，不好dq，所以我得去调研一下fp8里面对static per token quant的支持情况
    * TODO:dynamic channel wise weight/activation quant + scaled mm可行。见https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/ptpc_fp8.py
    * sq跑qwen2总是胡言乱语，问题出在对act的quant上面，这里无论使用static per tensor还是dyn per tensor/per channel都是胡言乱语，后续用opt试一下。quant sq qwen2应该是没有问题的，因为保持act为bf16就是正常输出
    * quant没有问题
* 以上
    * 已验证AWQ量化出来的模型，在runtime也可以run
        * 遇到的bug：把apply quant阶段的pesudo quant改成了real quant（结果real quant本质是发生在linear_awq.py，所以这里的修改是个多余操作），造成runtime过程中layer1的FA处输出nan，追溯bug发现是layer0的mlp处造成，后续使用autoAWQ量化一个模型出来试一下我们的runtime发现ok，于是定位到是quant有问题，最终rootcause，
        * 已解释linear awq.py处为什么要scales * zeros