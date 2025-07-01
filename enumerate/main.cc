#include <cstdlib>
#include <iostream>
#include <string_view>

#include <alpaka/alpaka.hpp>

#include "backend.h"
#include "config.h"

int main(int argc, const char* argv[]) {
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    if (arg == "-v" or arg == "--verbose") {
      verbose = true;
      continue;
    }
    if (arg == "-h" or arg == "--help") {
      std::string_view name{argv[0]};
      std::cout << "Usage:\n";
      std::cout << "  " << name << " [-v|--verbose]\n";
      std::cout << "  " << name << " [-h|--help]\n";
      exit(EXIT_SUCCESS);
    }
  }

  // the host platform always has a single device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);

  std::cout << "Host platform: " << alpaka::core::demangled<HostPlatform> << '\n';
  std::cout << "Found 1 device:\n";
  std::cout << "  - " << alpaka::getName(host) << "\n\n";

  alpaka_serial_sync::enumerate(verbose);
  alpaka_threads_sync::enumerate(verbose);
  alpaka_tbb_sync::enumerate(verbose);
  alpaka_cuda_async::enumerate(verbose);
}
