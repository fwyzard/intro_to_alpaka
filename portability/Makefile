.PHONY: all clean

all: 00_hello_world 01_hello_world

clean:
	rm -f 00_hello_world 01_hello_world

00_hello_world: 00_hello_world.cc
	g++ -std=c++20 -O2 -g 00_hello_world.cc -o 00_hello_world

01_hello_world: 01_hello_world.cu
	nvcc -std=c++20 -O2 -g 01_hello_world.cu -o 01_hello_world
