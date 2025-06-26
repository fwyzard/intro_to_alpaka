.PHONY: all clean

TARGETS := 00_enumerate_cpu 00_enumerate_cuda 01_blocking_queue_cpu 02_nonblocking_queue_cpu 03_memory_cpu 03_memory_cuda 04_views_cpu 04_views_cuda 05_kernel_cpu 05_kernel_cuda

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

# NVIDIA CUDA
CUDA_BASE   ?= /usr/local/cuda

# ROCM_BASE
ROCM_BASE   ?= /opt/rocm

# Intel TBB library
TBB_BASE    ?= /usr

# Alpaka library
ALPAKA_BASE ?= /afs/hep.wisc.edu/home/abocci/public/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc
HIPCC := $(ROCM_BASE)/bin/hipcc

CUDA_ARCH      := sm_87
ROCM_ARCH      := gfx1100

CXXFLAGS       := -std=c++20 -O2 -g -I$(ALPAKA_BASE)/include
CXX_HOST_FLAGS := -pthread
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) --extended-lambda --expt-relaxed-constexpr
CXX_ROCM_FLAGS := --offload-arch=$(ROCM_ARCH)

%_cpu: %.cc
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< -o $@

%_mt: %.cc
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED $< -o $@

%_tbb: %.cc
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) -I$(TBB_BASE)/include -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $< -L$(TBB_BASE)/lib -ltbb -o $@

%_cuda: %.cc
	$(NVCC) -x cu -ccbin $(CXX) $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' -DALPAKA_ACC_GPU_CUDA_ENABLED $< -o $@

%_hip: %.cc
	$(HIPCC) $(CXXFLAGS) $(CXX_HOST_FLAGS) $(CXX_ROCM_FLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED $< -o $@

