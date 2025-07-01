#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "backend.h"
#include "config.h"

namespace {
  template <typename T>
  std::ostream& operator<<(std::ostream& out, std::vector<T> const& v) {
    if (v.empty()) {
      out << "{}";
      return out;
    }
    out << "{ " << v[0];
    for (typename std::vector<T>::size_type i = 1; i < v.size(); ++i) {
      out << ", " << v[i];
    }
    out << " }";
    return out;
  }
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void enumerate(bool verbose) {
    // get all the devices on the accelerator platform
    Platform platform;
    std::vector<Device> devices = alpaka::getDevs(platform);

    std::cout << "Accelerator platform: " << alpaka::core::demangled<Platform> << '\n';
    std::cout << "Found " << devices.size() << " device(s):\n";
    for (auto const& device : devices) {
      std::cout << "  - " << alpaka::getName(device) << '\n';
      if (verbose) {
        std::cout << "    - Accelerator name: " << alpaka::core::demangled<Acc3D> << '\n';
        auto const props = alpaka::getAccDevProps<Acc3D>(device);
        auto const globalMem = alpaka::getMemBytes(device);
        auto const freeMem = alpaka::getFreeMemBytes(device);
        std::cout << "        number of multi-processors:           " << props.m_multiProcessorCount << '\n';
        std::cout << "        global memory free / total (bytes):   " << freeMem << " / " << globalMem << '\n';
        std::cout << "        shared memory per block (bytes):      " << props.m_sharedMemSizeBytes << '\n';
        std::cout << "        max blocks per grid (z, y, x):        " << props.m_gridBlockExtentMax << '\n';
        std::cout << "        max threads per block (z, y, x):      " << props.m_blockThreadExtentMax << '\n';
        std::cout << "        max elements per thread (z, y, x):    " << props.m_threadElemExtentMax << '\n';
        std::cout << "        max number of blocks per grid:        " << props.m_gridBlockCountMax << '\n';
        std::cout << "        max number of threads per block:      " << props.m_blockThreadCountMax << '\n';
        std::cout << "        max number of elements per thread:    " << props.m_threadElemCountMax << '\n';
        std::cout << "        supported warp sizes:                 " << getWarpSizes(device) << '\n';
        std::cout << "        preferred warp size:                  " << getPreferredWarpSize(device) << '\n';
      }
    }
    std::cout << '\n';
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
