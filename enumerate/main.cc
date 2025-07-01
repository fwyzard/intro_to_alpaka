#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "backend.h"
#include "config.h"

int main() {
  // the host platform always has a single device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);

  std::cout << "Host platform: " << alpaka::core::demangled<HostPlatform> << '\n';
  std::cout << "Found 1 device:\n";
  std::cout << "  - " << alpaka::getName(host) << "\n\n";

  alpaka_serial_sync::enumerate();
  //alpaka_threads_sync::enumerate();
  //alpaka_tbb_sync::enumerate();
  alpaka_cuda_async::enumerate();
  alpaka_rocm_async::enumerate();
  //alpaka_intelgpu_async::enumerate();
}
