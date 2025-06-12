* fp8 dyn
    1. torch.nn.Parameter全是fp32存储，切记！！在fwd的时候需要把weight to(fp8)再做gemm，但是在量化工具部分，这里没有执行fwd，只执行了linear的from linear和init，具体还需要在runtime验证
    2. 把per tensor设置为quant config