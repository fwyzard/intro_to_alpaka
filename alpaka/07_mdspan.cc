/*
 * g++ -std=c++17 -O2 -g -I$ALPAKA_BASE/include -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED 05_kernel.cc -o 05_kernel_cpu
 * nvcc -x cu -std=c++17 -O2 -g --expt-relaxed-constexpr -I$ALPAKA_BASE/include -DALPAKA_ACC_GPU_CUDA_ENABLED 05_kernel.cc -o 05_kernel_cuda
 */

#include <cassert>
#include <cstdio>
#include <random>

#include <experimental/mdspan>

#include <alpaka/alpaka.hpp>

#include "config.h"
#include "WorkDiv.hpp"

struct VectorAddKernelMD {
  /*
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                std::experimental::mdspan<const T, std::experimental::dextents<Idx, alpaka::Dim<TAcc>::value>> in1,
                                std::experimental::mdspan<const T, std::experimental::dextents<Idx, alpaka::Dim<TAcc>::value>> in2,
                                std::experimental::mdspan<T, std::experimental::dextents<Idx, alpaka::Dim<TAcc>::value>> out,
  */
  template <typename TAcc, typename TIn, typename TOut>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, TIn in1, TIn in2, TOut out, Vec3D size) const {
    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
      auto const index = alpaka::toArray(ndindex);
      out[index] = in1[index] + in2[index];
    }
  }
};

void testVectorAddKernelMD(Host host, Platform platform, Device device) {
  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // 3-dimensional and linearised buffer size
  constexpr Vec3D size = {50, 125, 16};

  // allocate input and output host buffers in pinned memory accessible by the Platform devices
  auto in1_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  auto in2_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  auto out_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);

  // access the 3D buffers using the mdspan interface
  auto in1_h_md = alpaka::experimental::getMdSpan(in1_h);
  auto in2_h_md = alpaka::experimental::getMdSpan(in2_h);
  auto out_h_md = alpaka::experimental::getMdSpan(out_h);

  // fill the input buffers with random data, and the output buffer with zeros
  for (uint32_t i = 0; i < size[0]; ++i) {
    for (uint32_t j = 0; j < size[1]; ++j) {
      for (uint32_t k = 0; k < size[2]; ++k) {
        in1_h_md(i, j, k) = dist(rand);
        in2_h_md(i, j, k) = dist(rand);
        out_h_md(i, j, k) = 0.;
      }
    }
  }

  // run the test on the given device
  auto queue = Queue{device};

  // allocate input and output buffers on the device
  auto in1_d = alpaka::allocBuf<float, uint32_t>(device, size);
  auto in2_d = alpaka::allocBuf<float, uint32_t>(device, size);
  auto out_d = alpaka::allocBuf<float, uint32_t>(device, size);

  // copy the input data to the device; the size is known from the buffer objects
  alpaka::memcpy(queue, in1_d, in1_h);
  alpaka::memcpy(queue, in2_d, in2_h);

  // fill the output buffer with zeros; the size is known from the buffer objects
  alpaka::fill(queue, out_d, 0.f);

  // launch the 3-dimensional kernel
  auto div = makeWorkDiv<Acc3D>({5, 5, 1}, {4, 4, 4});
  std::cout << "Testing VectorAddKernelMD with mdspan accessors with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc3D>(queue,
                      div,
                      VectorAddKernelMD{},
                      alpaka::experimental::getMdSpan(in1_d),
                      alpaka::experimental::getMdSpan(in2_d),
                      alpaka::experimental::getMdSpan(out_d),
                      size);

  // copy the results from the device to the host
  alpaka::memcpy(queue, out_h, out_d);

  // wait for all the operations to complete
  alpaka::wait(queue);

  // check the results
  for (uint32_t i = 0; i < size[0]; ++i) {
    for (uint32_t j = 0; j < size[1]; ++j) {
      for (uint32_t k = 0; k < size[2]; ++k) {
        float sum = in1_h_md(i, j, k) + in2_h_md(i, j, k);
        assert(out_h_md(i, j, k) < sum + epsilon);
        assert(out_h_md(i, j, k) > sum - epsilon);
      }
    }
  }
  std::cout << "success\n";
}

int main() {
  // initialise the accelerator platform
  Platform platform;

  // require at least one device
  std::uint32_t n = alpaka::getDevCount(platform);
  if (n == 0) {
    exit(EXIT_FAILURE);
  }

  // use the single host device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);
  std::cout << "Host:   " << alpaka::getName(host) << '\n';

  // use the first device
  Device device = alpaka::getDevByIdx(platform, 0u);
  std::cout << "Device: " << alpaka::getName(device) << '\n';

  testVectorAddKernelMD(host, platform, device);
}
