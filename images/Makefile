.PHONY: all clean

CXX := g++

all: test

clean:
	rm -f test

stb:
	git clone https://github.com/nothings/stb.git

fmt:
	git clone https://github.com/fmtlib/fmt.git

libsixel:
	git clone https://github.com/saitoha/libsixel.git build/libsixel && cd build/libsixel && ./configure --without-libcurl --without-jpeg --without-png --without-pkgconfigdir --without-bashcompletiondir --without-zshcompletiondir --disable-python --prefix=$(shell realpath libsixel) && make -j`nproc` install && cd ../../ && ln -s libsixel/lib/libsixel.so.1.0.6 libsixel.so.1 && rm -rf build

test: test.cc Makefile stb fmt libsixel
	$(CXX) -std=c++20 -O3 -g -Istb -Ifmt/include -Ilibsixel/include -Wall -march=native $< -Llibsixel/lib -lsixel -o $@

