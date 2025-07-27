.PHONY: all clean

all: enumerate

clean:
	rm -f enumerate *.o *.so

# NVIDIA CUDA
CUDA_BASE   ?= /usr/local/cuda

# AMD HIP/ROCm
ROCM_BASE   ?= /opt/rocm

# Intel oneAPI
ONEAPI_BASE ?= /opt/intel/oneapi

# Intel TBB library
TBB_BASE    ?= /usr

# Alpaka library
ALPAKA_BASE ?= $(HOME)/src/alpaka-group/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc
HIPCC := $(ROCM_BASE)/bin/hipcc
ICPX := $(ONEAPI_BASE)/compiler/latest/bin/icpx

CUDA_ARCH      := sm_86
ROCM_ARCH      := gfx1100
SYCL_GPU_ARCH  := intel_gpu_tgllp

CXXFLAGS       := -std=c++20 -O2 -g -I$(ALPAKA_BASE)/include -DALPAKA_HAS_STD_ATOMIC_REF
CXX_HOST_FLAGS := -fPIC -pthread
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr
CXX_ROCM_FLAGS := --gcc-toolchain=$(shell $(CXX) -v 2>&1 | grep Configured | xargs -n1 echo | grep '^--prefix' | cut -d= -f2-) --gcc-triple=$(shell $(CXX) -dumpmachine) --offload-arch=$(ROCM_ARCH)
CXX_SYCL_CPU_FLAGS := --gcc-toolchain=$(shell $(CXX) -v 2>&1 | grep Configured | xargs -n1 echo | grep '^--prefix' | cut -d= -f2-) --gcc-triple=$(shell $(CXX) -dumpmachine) -fsycl -fsycl-targets=$(SYCL_GPU_ARCH) -I$(ONEAPI_BASE)/compiler/latest/include -I$(ONEAPI_BASE)/dpl/latest/include
CXX_SYCL_GPU_FLAGS := --gcc-toolchain=$(shell $(CXX) -v 2>&1 | grep Configured | xargs -n1 echo | grep '^--prefix' | cut -d= -f2-) --gcc-triple=$(shell $(CXX) -dumpmachine) -fsycl -fsycl-targets=spir64_x86_64 -I$(ONEAPI_BASE)/compiler/latest/include -I$(ONEAPI_BASE)/dpl/latest/include

%.o: %.cc Makefile
	$(CXX) -c $(CXXFLAGS) $(CXX_HOST_FLAGS) $< -o $@

lib%_cpu.so: %.cc Makefile
	$(CXX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< -o $@

lib%_mt.so: %.cc Makefile
	$(CXX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED $< -o $@

lib%_tbb.so: %.cc Makefile
	$(CXX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) -I$(TBB_BASE)/include -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $< -L$(TBB_BASE)/lib -Wl,-rpath=$(TBB_BASE)/lib -ltbb -o $@

lib%_cuda.so: %.cc Makefile
	$(NVCC) -x cu -ccbin $(CXX) -shared $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' -DALPAKA_ACC_GPU_CUDA_ENABLED $< -o $@

lib%_hip.so: %.cc Makefile
	$(HIPCC) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) $(CXX_ROCM_FLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED $< -o $@

lib%_syclcpu.so: %.cc Makefile
	. $(ONEAPI_BASE)/setvars.sh > /dev/null 2>&1 || true; $(ICPX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) $(CXX_SYCL_CPU_FLAGS) -DALPAKA_ACC_SYCL_ENABLED -DALPAKA_SYCL_ONEAPI_CPU $< -L$(ONEAPI_BASE)/compiler/latest/lib -Wl,-rpath=$(ONEAPI_BASE)/compiler/latest/lib -o $@

lib%_syclgpu.so: %.cc Makefile
	. $(ONEAPI_BASE)/setvars.sh > /dev/null 2>&1 || true; $(ICPX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) $(CXX_SYCL_GPU_FLAGS) -DALPAKA_ACC_SYCL_ENABLED -DALPAKA_SYCL_ONEAPI_GPU $< -L$(ONEAPI_BASE)/compiler/latest/lib -Wl,-rpath=$(ONEAPI_BASE)/lib -o $@

enumerate: main.o libbackend_cpu.so libbackend_mt.so libbackend_tbb.so libbackend_cuda.so libbackend_hip.so libbackend_syclcpu.so libbackend_syclgpu.so Makefile
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) $< -L. -lbackend_cpu -lbackend_mt -lbackend_tbb -Wl,-rpath-link=$(TBB_BASE)/lib -lbackend_cuda -lbackend_hip -lbackend_syclcpu -lbackend_syclgpu -Wl,-rpath-link=$(ONEAPI_BASE)/compiler/latest/lib -o $@
