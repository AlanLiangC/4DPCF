# JIT
from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["open4dpcf/ops/dvr/dvr.cpp", "open4dpcf/ops/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
