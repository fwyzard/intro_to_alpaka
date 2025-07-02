/*
Copyright (C) 2024 Andrea Bocci
SPDX-License-Identifier: GNU General Public License v3.0 or later

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef __linux__
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define FMT_HEADER_ONLY
#include <fmt/color.h>
#include <fmt/core.h>

#include <sixel.h>

#include "WorkDiv.hpp"
#include "config.h"
#include <alpaka/alpaka.hpp>

using namespace std::literals;

// global objects for the host and device platforms
HostPlatform host_platform;
Host host = alpaka::getDevByIdx(host_platform, 0u);

Platform platform;
Device device = alpaka::getDevByIdx(platform, 0u);

struct Image {
  unsigned char *data_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;

  Image() {}

  Image(std::string const &filename) { open(filename); }

  Image(int width, int height, int channels)
      : width_(width), height_(height), channels_(channels) {
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char *>(stbi__malloc(size));
    std::memset(data_, 0x00, size);
  }

  ~Image() { close(); }

  // copy constructor
  Image(Image const &img)
      : width_(img.width_), height_(img.height_), channels_(img.channels_) {
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char *>(stbi__malloc(size));
    std::memcpy(data_, img.data_, size);
  }

  // copy assignment
  Image &operator=(Image const &img) {
    // avoid self-copies
    if (&img == this) {
      return *this;
    }

    // free any existing image data
    close();

    width_ = img.width_;
    height_ = img.height_;
    channels_ = img.channels_;
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char *>(stbi__malloc(size));
    std::memcpy(data_, img.data_, size);

    return *this;
  }

  // move constructor
  Image(Image &&img)
      : data_(img.data_), width_(img.width_), height_(img.height_),
        channels_(img.channels_) {
    // take owndership of the image data
    img.data_ = nullptr;
  }

  // move assignment
  Image &operator=(Image &&img) {
    // avoid self-moves
    if (&img == this) {
      return *this;
    }

    // free any existing image data
    close();

    // copy the image properties
    width_ = img.width_;
    height_ = img.height_;
    channels_ = img.channels_;

    // take owndership of the image data
    data_ = img.data_;
    img.data_ = nullptr;

    return *this;
  }

  void open(std::string const &filename) {
    data_ = stbi_load(filename.c_str(), &width_, &height_, &channels_, 0);
    if (data_ == nullptr) {
      throw std::runtime_error("Failed to load "s + filename);
    }
    std::cout << "Loaded image with " << width_ << " x " << height_
              << " pixels and " << channels_ << " channels from " << filename
              << '\n';
  }

  void write(std::string const &filename) {
    if (filename.ends_with(".png")) {
      int status = stbi_write_png(filename.c_str(), width_, height_, channels_,
                                  data_, 0);
      if (status == 0) {
        throw std::runtime_error("Error while writing PNG file "s + filename);
      }
    } else if (filename.ends_with(".jpg") or filename.ends_with(".jpeg")) {
      int status = stbi_write_jpg(filename.c_str(), width_, height_, channels_,
                                  data_, 95);
      if (status == 0) {
        throw std::runtime_error("Error while writing JPEG file "s + filename);
      }
    } else {
      throw std::runtime_error("File format "s + filename + "not supported"s);
    }
  }

  void close() {
    if (data_ != nullptr) {
      stbi_image_free(data_);
    }
    data_ = nullptr;
  }

  static int sixel_write(char *data, int size, void *priv) {
    return fwrite(data, 1, size, (FILE *)priv);
  }

  // show an image on the terminal, using up to max_width columns (with one
  // block per column) and up to max_height lines (with two blocks per line)
  void show(int max_width, int max_height) {
    if (data_ == nullptr) {
      return;
    }

    /*
    // find the best size given the max width and height and the image aspect
    ratio int width, height; if (width_ * max_height > height_ * max_width) {
      width = max_width;
      height = max_width * height_ / width_;
    } else {
      width = max_height * width_ / height_;
      height = max_height;
    }
    */

    sixel_output_t *output = nullptr;
    auto status = sixel_output_new(&output, sixel_write, stdout, nullptr);
    if (SIXEL_FAILED(status))
      exit(EXIT_FAILURE);

    sixel_dither_t *dither = sixel_dither_get(SIXEL_BUILTIN_XTERM256);
    if (channels_ == 1) {
      sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_G8);
    } else if (channels_ == 3) {
      sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_RGB888);
    } else if (channels_ == 4) {
      sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_RGBA8888);
    }

    status = sixel_encode(data_, width_, height_, 0, dither, output);
    if (SIXEL_FAILED(status))
      exit(EXIT_FAILURE);
  }

  auto view() {
    return alpaka::createView(host, data_, Vec1D{width_ * height_ * channels_});
  }

  auto view() const {
    return alpaka::ViewConst(
        alpaka::createView(host, data_, Vec1D{width_ * height_ * channels_}));
  }

  auto span() {
    return std::span<unsigned char>(data_, width_ * height_ * channels_);
  }

  auto span() const {
    return std::span<const unsigned char>(data_, width_ * height_ * channels_);
  }
};

struct ImageView {
  std::span<unsigned char> data_;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
};

struct ImageViewConst {
  std::span<const unsigned char> data_;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
};

struct ImageDevice {
  alpaka::Buf<Device, unsigned char, Dim1D, uint32_t> data_;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;

  ImageDevice(Queue &queue, int width, int height, int channels)
      : data_{alpaka::allocAsyncBuf<unsigned char, uint32_t>(
            queue, Vec1D{width * height * channels})},
        width_{width}, height_{height}, channels_{channels} {}

  auto view() { return data_; }

  auto view() const { return alpaka::ViewConst(data_); }

  auto span() {
    return std::span<unsigned char>(data_.data(), width_ * height_ * channels_);
  }

  auto span() const {
    return std::span<const unsigned char>(data_.data(),
                                          width_ * height_ * channels_);
  }

  ImageView imageView() {
    return ImageView{span(), width_, height_, channels_};
  }

  ImageViewConst imageView() const {
    return ImageViewConst{span(), width_, height_, channels_};
  }
};

void copy_to_device(Queue &queue, ImageDevice &dst, Image &src) {
  // check that the source and destination images have the same properties
  assert(dst.width_ == src.width_);
  assert(dst.height_ == src.height_);
  assert(dst.channels_ == src.channels_);
  // copy the image data
  alpaka::memcpy(queue, dst.view(), src.view());
}

void copy_to_host(Queue &queue, Image &dst, ImageDevice &src) {
  // check that the source and destination images have the same properties
  assert(dst.width_ == src.width_);
  assert(dst.height_ == src.height_);
  assert(dst.channels_ == src.channels_);
  // copy the image data
  alpaka::memcpy(queue, dst.view(), src.view());
}

bool verbose = false;

// make a scaled copy of an image
struct Scale {
  ALPAKA_FN_ACC
  void operator()(Acc2D const &acc, const ImageViewConst src,
                  ImageView out) const {

    for (int y : alpaka::uniformElementsAlongY(acc, out.height_)) {
      // map the row of the scaled image to the nearest rows of the original
      // image
      float yp = static_cast<float>(y) * src.height_ / out.height_;
      int y0 = std::clamp(static_cast<int>(std::floor(yp)), 0, src.height_ - 1);
      int y1 = std::clamp(static_cast<int>(std::ceil(yp)), 0, src.height_ - 1);

      // interpolate between y0 and y1
      float wy0 = yp - y0;
      float wy1 = y1 - yp;
      // if the new y coorindate maps to an integer coordinate in the original
      // image, use a fake distance from identical values corresponding to it
      if (y0 == y1) {
        wy0 = 1.f;
        wy1 = 1.f;
      }
      float dy = wy0 + wy1;

      for (int x : alpaka::uniformElementsAlongX(acc, out.width_)) {
        int p = (y * out.width_ + x) * out.channels_;

        // map the column of the scaled image to the nearest columns of the
        // original image
        float xp = static_cast<float>(x) * src.width_ / out.width_;
        int x0 =
            std::clamp(static_cast<int>(std::floor(xp)), 0, src.width_ - 1);
        int x1 = std::clamp(static_cast<int>(std::ceil(xp)), 0, src.width_ - 1);

        // interpolate between x0 and x1
        float wx0 = xp - x0;
        float wx1 = x1 - xp;
        // if the new x coordinate maps to an integer coordinate in the original
        // image, use a fake distance from identical values corresponding to it
        if (x0 == x1) {
          wx0 = 1.f;
          wx1 = 1.f;
        }
        float dx = wx0 + wx1;

        // bi-linear interpolation of all channels
        int p00 = (y0 * src.width_ + x0) * src.channels_;
        int p10 = (y1 * src.width_ + x0) * src.channels_;
        int p01 = (y0 * src.width_ + x1) * src.channels_;
        int p11 = (y1 * src.width_ + x1) * src.channels_;

        for (int c = 0; c < src.channels_; ++c) {
          out.data_[p + c] = static_cast<unsigned char>(std::round(
              (src.data_[p00 + c] * wx1 * wy1 + src.data_[p10 + c] * wx1 * wy0 +
               src.data_[p01 + c] * wx0 * wy1 +
               src.data_[p11 + c] * wx0 * wy0) /
              (dx * dy)));
        }
      }
    }
  }
};

ImageDevice scale(Queue &queue, ImageDevice const &src, int width, int height) {
  if (width == src.width_ and height == src.height_) {
    // if the dimensions are the same, return the same image
    return src;
  }

  // create a new image
  ImageDevice out(queue, width, height, src.channels_);

  auto start = std::chrono::steady_clock::now();

  auto grid = makeWorkDiv<Acc2D>(Vec2D{16, 16}, Vec2D{16, 16});
  alpaka::exec<Acc2D>(queue, grid, Scale{}, src.imageView(), out.imageView());

  auto finish = std::chrono::steady_clock::now();
  float ms =
      std::chrono::duration_cast<std::chrono::duration<float>>(finish - start)
          .count() *
      1000.f;
  if (verbose) {
    std::cerr << fmt::format("scale:      {:6.2f}", ms) << " ms\n";
  }

  return out;
}

// copy a source image into a target image, cropping any parts that fall outside
// the target image
void write_to(Image const &src, Image &dst, int x, int y) {
  // copying to an image with a different number of channels is not supported
  assert(src.channels_ == dst.channels_);

  // the whole source image would fall outside of the target image along the X
  // axis
  if ((x + src.width_ < 0) or (x >= dst.width_)) {
    return;
  }

  // the whole source image would fall outside of the target image along the Y
  // axis
  if ((y + src.height_ < 0) or (y >= dst.height_)) {
    return;
  }

  // find the valid range for the overlapping part of the images along the X and
  // Y axes
  int src_x_from = std::max(0, -x);
  int src_x_to = std::min(src.width_, dst.width_ - x);
  int dst_x_from = std::max(0, x);
  // int dst_x_to   = std::min(src.width_ + x, dst.width_);
  int x_width = src_x_to - src_x_from;

  int src_y_from = std::max(0, -y);
  int src_y_to = std::min(src.height_, dst.height_ - y);
  int dst_y_from = std::max(0, y);
  // int dst_y_to   = std::min(src.height_ + y, dst.height_);
  int y_height = src_y_to - src_y_from;

  auto start = std::chrono::steady_clock::now();

  for (int y = 0; y < y_height; ++y) {
    int src_p = ((src_y_from + y) * src.width_ + src_x_from) * src.channels_;
    int dst_p = ((dst_y_from + y) * dst.width_ + dst_x_from) * dst.channels_;
    std::memcpy(dst.data_ + dst_p, src.data_ + src_p, x_width * src.channels_);
  }

  auto finish = std::chrono::steady_clock::now();
  float ms =
      std::chrono::duration_cast<std::chrono::duration<float>>(finish - start)
          .count() *
      1000.f;
  if (verbose) {
    std::cerr << fmt::format("write_to:   {:6.2f}", ms) << " ms\n";
  }
}

struct Grayscale {
  ALPAKA_FN_ACC
  void operator()(Acc2D const &acc, ImageView img) const {
    for (auto idx :
         alpaka::uniformElementsND(acc, Vec2D{img.height_, img.width_})) {
      int p = (idx.y() * img.width_ + idx.x()) * img.channels_;
      int r = img.data_[p];
      int g = img.data_[p + 1];
      int b = img.data_[p + 2];
      // NTSC values for RGB to grayscale conversion
      int y = (299 * r + 587 * g + 114 * b) / 1000;
      img.data_[p] = y;
      img.data_[p + 1] = y;
      img.data_[p + 2] = y;
    }
  }
};

// convert an image to grayscale
ImageDevice grayscale(Queue &queue, ImageDevice const &src) {
  // non-RGB images are not supported
  assert(src.channels_ >= 3);

  auto start = std::chrono::steady_clock::now();

  ImageDevice dst(queue, src.width_, src.height_, src.channels_);
  alpaka::memcpy(queue, dst.view(), src.view());
  auto grid = makeWorkDiv<Acc2D>(Vec2D{16, 16}, Vec2D{16, 16});
  alpaka::exec<Acc2D>(queue, grid, Grayscale{}, dst.imageView());

  auto finish = std::chrono::steady_clock::now();
  float ms =
      std::chrono::duration_cast<std::chrono::duration<float>>(finish - start)
          .count() *
      1000.f;
  if (verbose) {
    std::cerr << fmt::format("grayscale:  {:6.2f}", ms) << " ms\n";
  }

  return dst;
}

// apply an RGB tint to an image
struct Tint {
  ALPAKA_FN_ACC
  void operator()(Acc2D const &acc, ImageView img, int r, int g, int b) const {
    for (auto idx :
         alpaka::uniformElementsND(acc, Vec2D{img.height_, img.width_})) {
      int p = (idx.y() * img.width_ + idx.x()) * img.channels_;
      int r0 = img.data_[p];
      int g0 = img.data_[p + 1];
      int b0 = img.data_[p + 2];
      img.data_[p] = r0 * r / 255;
      img.data_[p + 1] = g0 * g / 255;
      img.data_[p + 2] = b0 * b / 255;
    }
  }
};

// apply an RGB tint to an image
ImageDevice tint(Queue &queue, ImageDevice const &src, int r, int g, int b) {
  // non-RGB images are not supported
  assert(src.channels_ >= 3);

  auto start = std::chrono::steady_clock::now();

  ImageDevice dst(queue, src.width_, src.height_, src.channels_);
  alpaka::memcpy(queue, dst.view(), src.view());
  auto grid = makeWorkDiv<Acc2D>(Vec2D{16, 16}, Vec2D{16, 16});
  alpaka::exec<Acc2D>(queue, grid, Tint{}, dst.imageView(), r, g, b);

  auto finish = std::chrono::steady_clock::now();
  float ms =
      std::chrono::duration_cast<std::chrono::duration<float>>(finish - start)
          .count() *
      1000.f;
  if (verbose) {
    std::cerr << fmt::format("tint:       {:6.2f}", ms) << " ms\n";
  }

  return dst;
}

int main(int argc, const char *argv[]) {
  const char *verbose_env = std::getenv("VERBOSE");
  if (verbose_env != nullptr and std::strlen(verbose_env) != 0) {
    verbose = true;
  }

  std::vector<std::string> files;
  if (argc == 1) {
    // no arguments, use a single default image
    files = {"image.jpg"s};
  } else {
    files.reserve(argc - 1);
    for (int i = 1; i < argc; ++i) {
      files.emplace_back(argv[i]);
    }
  }

  int rows = 80;
  int columns = 80;
#if defined(__linux__) && defined(TIOCGWINSZ)
  if (isatty(STDOUT_FILENO)) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    if (w.ws_row > 1 and w.ws_col > 1) {
      rows = w.ws_row - 1;
      columns = w.ws_col - 1;
    }
  }
#endif

  // use the single host device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);
  std::cout << "Host:   " << alpaka::getName(host) << '\n';

  // initialise the accelerator platform and require at least one device
  Platform platform;
  std::uint32_t n = alpaka::getDevCount(platform);
  if (n == 0) {
    exit(EXIT_FAILURE);
  }
  std::cout << "Device: " << alpaka::getName(device) << '\n';
  Queue queue{device};

  std::vector<Image> images;
  images.resize(files.size());
  for (unsigned int i = 0; i < files.size(); ++i) {
    auto &img = images[i];
    img.open(files[i]);
    img.show(columns, rows); // FIXME columns and rows are currently ignored

    ImageDevice img_d(queue, img.width_, img.height_, img.channels_);
    copy_to_device(queue, img_d, img);
    ImageDevice small_d =
        scale(queue, img_d, img_d.width_ * 0.5, img_d.height_ * 0.5);
    ImageDevice gray_d = grayscale(queue, small_d);
    ImageDevice tone1_d = tint(queue, gray_d, 168, 56, 172); // purple-ish
    ImageDevice tone2_d = tint(queue, gray_d, 100, 143, 47); // green-ish
    ImageDevice tone3_d = tint(queue, gray_d, 255, 162, 36); // gold-ish

    Image gray(gray_d.width_, gray_d.height_, gray_d.channels_);
    Image tone1(tone1_d.width_, tone1_d.height_, tone1_d.channels_);
    Image tone2(tone2_d.width_, tone2_d.height_, tone2_d.channels_);
    Image tone3(tone3_d.width_, tone3_d.height_, tone3_d.channels_);

    copy_to_host(queue, gray, gray_d);
    copy_to_host(queue, tone1, tone1_d);
    copy_to_host(queue, tone2, tone2_d);
    copy_to_host(queue, tone3, tone3_d);
    alpaka::wait(queue);

    Image out(img.width_, img.height_, img.channels_);
    write_to(tone1, out, 0, 0);
    write_to(tone2, out, img.width_ * 0.5, 0);
    write_to(tone3, out, 0, img.height_ * 0.5);
    write_to(gray, out, img.width_ * 0.5, img.height_ * 0.5);

    std::cout << '\n';
    out.show(columns, rows);
    out.write(fmt::format("out{:02d}.jpg", i));
  }

  return 0;
}
