mainly for quantization only, output a quantized model, then input the quantized model into runtime to run with custom low-bit gemm kernel.

* pre-requirements by pip install: transformers==4.46.0, zstandard, datasets==3.6.0, accelerate
* danie torch docker里面datasets可以到3.6，但是danie torch2504不行，因为依赖dill版本小于0.3.9，后者dill版本无法降级到0.3.8