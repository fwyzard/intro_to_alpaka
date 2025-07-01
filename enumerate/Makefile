.PHONY: all clean

all: enumerate

clean:
	rm -f enumerate *.o *.so

# NVIDIA CUDA
CUDA_BASE   ?= /usr/local/cuda

# ROCM_BASE
ROCM_BASE   ?= /opt/rocm

# Intel TBB library
TBB_BASE    ?= /usr

# Alpaka library
ALPAKA_BASE ?= $(HOME)/test/alpaka_enumerate/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc
HIPCC := $(ROCM_BASE)/bin/hipcc

CUDA_ARCH      := sm_89
ROCM_ARCH      := gfx1100

CXXFLAGS       := -std=c++20 -O2 -g -I$(ALPAKA_BASE)/include -DALPAKA_HAS_STD_ATOMIC_REF
CXX_HOST_FLAGS := -fPIC -pthread
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr
CXX_ROCM_FLAGS := --gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr --gcc-triple=x86_64-redhat-linux --offload-arch=$(ROCM_ARCH)


%.o: %.cc Makefile
	$(CXX) -c $(CXXFLAGS) $(CXX_HOST_FLAGS) $< -o $@

lib%.cpu.so: %.cc Makefile
	$(CXX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< -o $@

lib%.mt.so: %.cc Makefile
	$(CXX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED $< -o $@

lib%.tbb.so: %.cc Makefile
	$(CXX) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) -I$(TBB_BASE)/include -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $< -L$(TBB_BASE)/lib -Wl,-rpath=$(TBB_BASE)/lib -ltbb -o $@

lib%.cuda.so: %.cc Makefile
	$(NVCC) -x cu -ccbin $(CXX) -shared $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' -DALPAKA_ACC_GPU_CUDA_ENABLED $< -o $@

lib%.hip.so: %.cc Makefile
	$(HIPCC) -shared $(CXXFLAGS) $(CXX_HOST_FLAGS) $(CXX_ROCM_FLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED $< -o $@

enumerate: main.o libbackend.cpu.so libbackend.mt.so libbackend.tbb.so libbackend.cuda.so libbackend.hip.so Makefile
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) $< -L. -lbackend.cpu -lbackend.mt -lbackend.tbb -Wl,-rpath-link=$(TBB_BASE)/lib -lbackend.cuda -o $@
