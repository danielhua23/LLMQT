from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

model_path = 'Qwen/Qwen2.5-14B-Instruct'
quant_path = 'Qwen2.5-14B-Instruct-fp8-static'
# quant_config = {"quant_method": "fp8_static_quant", "zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM" }
quant_config = {"quant_method": "fp8_static_quant"}

# Load model
model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)#, export_compatible=True表示fake quant

# Save quantized model
model.save_quantized(quant_path)# export_compatible=True的情况下，这个是fake quant model
tokenizer.save_pretrained(quant_path)
# export_compatible=True的情况下，运行下列两行保留real quantized model
# model.pack(...) # makes the model CUDA compat
# model.save_quantized(...)  # produces CUDA compat weights
print(f'Model is quantized and saved at "{quant_path}"')