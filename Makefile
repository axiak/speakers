CC = gcc
WFLAGS = -Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wcast-align \
         -Wstrict-prototypes -Wall -Wwrite-strings -Wcast-qual \
         -Wswitch-default -fPIC \
         -pedantic -std=c11 -pthread

SRC = ./src

OPTFLAGS = -O3 -ffast-math
DEBUG_FLAGS = -g -O0

LIBS = -lfftw3f -lportaudio -lm

ARCH_FLAGS = $(shell ./gen_arch_flags.sh)

all:
	cd libfilter; make all

clean:
	cd libfilter; make clean

install-deps-apt:
	apt-get install -y build-essential python python-pyaudio portaudio19-dev \
		libfftw3-dev libfftw3-bin libfftw3-single3 \
		python-pyalsa python-numpy python-scipy python-matplotlib python-matplotlib-data \
		ipython python-cffi python-leveldb libsndfile1-dev \
		python-virtualenv
