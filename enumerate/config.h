/* 
 * This file is part of CMSSW, see https://github.com/cms-sw/cmssw/ 
 *
 * Copyright 2023 CERN¹ for the benefit of the CMS Collaboration²
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * ¹ https://home.cern/
 * ² https://cms.cern/
 */

#ifndef config_h
#define config_h

#include <alpaka/alpaka.hpp>

// index type
using Idx = uint32_t;

// dimensions
using Dim0D = alpaka::DimInt<0u>;
using Dim1D = alpaka::DimInt<1u>;
using Dim2D = alpaka::DimInt<2u>;
using Dim3D = alpaka::DimInt<3u>;

// vectors
template <typename TDim>
using Vec = alpaka::Vec<TDim, uint32_t>;
using Scalar = Vec<Dim0D>;
using Vec1D = Vec<Dim1D>;
using Vec2D = Vec<Dim2D>;
using Vec3D = Vec<Dim3D>;

// work division
template <typename TDim>
using WorkDiv = alpaka::WorkDivMembers<TDim, uint32_t>;
using WorkDiv1D = WorkDiv<Dim1D>;
using WorkDiv2D = WorkDiv<Dim2D>;
using WorkDiv3D = WorkDiv<Dim3D>;

// host types
using Host = alpaka::DevCpu;
using HostPlatform = alpaka::PlatformCpu;

// different backends and device types

// FIXME
// replace
//    using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
// with
//    using Device = alpaka::Dev<Tag>
// one alpaka implements it

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
// NVIDIA CUDA backend
using Tag = alpaka::TagGpuCudaRt;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async

#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
// AMD HIP/ROCm backend
using Tag = alpaka::TagGpuHipRt;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_rocm_async

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
// CPU serial backend
using Tag = alpaka::TagCpuSerial;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
// CPU parallel backend using one CPU thread per device thread
using Tag = alpaka::TagCpuThreads;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_threads_sync

#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
// CPU parallel backend using one TBB task per block
using Tag = alpaka::TagCpuTbbBlocks;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_sync

#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)
// Intel CPU backend
using Tag = alpaka::TagCpuSycl;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_syclcpu_async

#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)
// Intel GPU backend
using Tag = alpaka::TagGpuSyclIntel;
using Device = alpaka::Dev<alpaka::TagToAcc<Tag, Dim1D, uint32_t>>;
using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_intelgpu_async

#else
// no backend specified
//#error Please define a single one of ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_GPU_HIP_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED, ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#endif //  defined(ALPAKA_ACC_...)

#if defined ALPAKA_ACCELERATOR_NAMESPACE

// common definitions, if any backend was specified
using Platform = alpaka::Platform<Device>;
using Event = alpaka::Event<Queue>;

template <typename TDim>
using Acc = alpaka::TagToAcc<Tag, TDim, uint32_t>;
using Acc1D = Acc<Dim1D>;
using Acc2D = Acc<Dim2D>;
using Acc3D = Acc<Dim3D>;

#endif  // defined ALPAKA_ACCELERATOR_NAMESPACE

#endif  // config_h
