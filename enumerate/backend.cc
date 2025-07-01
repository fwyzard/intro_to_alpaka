#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "backend.h"
#include "config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void enumerate() {
    // get all the devices on the accelerator platform
    Platform platform;
    std::vector<Device> devices = alpaka::getDevs(platform);

    std::cout << "Accelerator platform: " << alpaka::core::demangled<Platform> << '\n';
    std::cout << "Found " << devices.size() << " device(s):\n";
    for (auto const& device : devices)
      std::cout << "  - " << alpaka::getName(device) << '\n';
    std::cout << '\n';
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
