.PHONY: all clean

SUFFIX  := cpu mt tbb cuda hip
SOURCES := $(wildcard *.cc)
TARGETS := $(foreach @, $(SUFFIX), $(SOURCES:%.cc=%_$@))

all: cpu mt tbb cuda

cpu: $(SOURCES:%.cc=%_cpu)

mt:  $(SOURCES:%.cc=%_mt)

tbb: $(SOURCES:%.cc=%_tbb)

cuda: $(SOURCES:%.cc=%_cuda)

hip: $(SOURCES:%.cc=%_hip)

clean:
	rm -f $(TARGETS)

# NVIDIA CUDA
CUDA_BASE   ?= /usr/local/cuda

# ROCM_BASE
ROCM_BASE   ?= /opt/rocm

# Intel TBB library
TBB_BASE    ?= /usr

# mdspan library (included in c++23)
MDSPAN_BASE ?= /home/fwyzard/src/alpaka-group/mdspan

# Alpaka library
ALPAKA_BASE ?= /home/fwyzard/src/alpaka-group/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc
HIPCC := $(ROCM_BASE)/bin/hipcc

CUDA_ARCH      := sm_86
ROCM_ARCH      := gfx1100

CXXFLAGS       := -std=c++20 -O2 -g -I$(MDSPAN_BASE)/include -I$(ALPAKA_BASE)/include -DALPAKA_HAS_STD_ATOMIC_REF -DALPAKA_USE_MDSPAN
CXX_HOST_FLAGS := -pthread
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr
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

