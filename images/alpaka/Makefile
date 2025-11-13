.PHONY: all clean

SUFFIX  := cpu mt tbb cuda hip
SOURCES := test.cc
TARGETS := $(foreach @, $(SUFFIX), $(SOURCES:%.cc=%_$@))

all: cpu mt tbb cuda

cpu: $(SOURCES:%.cc=%_cpu)

mt:  $(SOURCES:%.cc=%_mt)

tbb: $(SOURCES:%.cc=%_tbb)

cuda: $(SOURCES:%.cc=%_cuda)

hip: $(SOURCES:%.cc=%_hip)

clean:
	rm -f $(TARGETS)

stb:
	git clone https://github.com/nothings/stb.git

fmt:
	git clone https://github.com/fmtlib/fmt.git

libsixel:
	git clone https://github.com/saitoha/libsixel.git build/libsixel && cd build/libsixel && ./configure --without-libcurl --without-jpeg --without-png --without-pkgconfigdir --without-bashcompletiondir --without-zshcompletiondir --disable-python --prefix=$(shell realpath libsixel) && make -j`nproc` install && cd ../../ && rm -rf build

# NVIDIA CUDA
CUDA_BASE   ?= /usr/local/cuda

# ROCM_BASE
ROCM_BASE   ?= /opt/rocm

# Intel TBB library
TBB_BASE    ?= /opt/intel/oneapi/tbb/latest

# mdspan library (included in c++23)
MDSPAN_BASE ?= $(HOME)/public/mdspan

# Alpaka library
ALPAKA_BASE ?= $(HOME)/public/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc
HIPCC := $(ROCM_BASE)/bin/hipcc

CUDA_ARCH      := sm_75
ROCM_ARCH      := gfx1100

CXXFLAGS       := -std=c++20 -O2 -g -I$(MDSPAN_BASE)/include -I$(ALPAKA_BASE)/include -Istb -Ifmt/include -Ilibsixel/include -DALPAKA_HAS_STD_ATOMIC_REF -DALPAKA_USE_MDSPAN
LDFLAGS        := -Llibsixel/lib -lsixel
CXX_HOST_FLAGS := -pthread
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr -diag-suppress 550 -diag-suppress 20208
CXX_ROCM_FLAGS := --offload-arch=$(ROCM_ARCH)

%_cpu: %.cc stb fmt libsixel
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< $(LDFLAGS) -o $@

%_mt: %.cc stb fmt libsixel
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED $< $(LDFLAGS) -o $@

%_tbb: %.cc stb fmt libsixel
	$(CXX) $(CXXFLAGS) $(CXX_HOST_FLAGS) -I$(TBB_BASE)/include -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $< $(LDFLAGS) -L$(TBB_BASE)/lib -ltbb -o $@

%_cuda: %.cc stb fmt libsixel
	$(NVCC) -x cu -ccbin $(CXX) $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' -DALPAKA_ACC_GPU_CUDA_ENABLED $< $(LDFLAGS) -o $@

%_hip: %.cc stb fmt libsixel
	$(HIPCC) $(CXXFLAGS) $(CXX_HOST_FLAGS) $(CXX_ROCM_FLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED $< $(LDFLAGS) -o $@

