CC = gcc
WFLAGS = -Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wcast-align \
         -Wstrict-prototypes -Wall -Wwrite-strings -Wcast-qual \
         -Wswitch-default -fPIC \
         -pedantic -std=c11 -pthread

OPTFLAGS = -O3 -ffast-math
DEBUG_FLAGS = -g -O0

LIBS = -lfftw3f -lportaudio -lm

ARCH_FLAGS = $(shell ./gen_arch_flags.sh)

all: libfilter

clean:
	rm -rfv target/

target:
	mkdir target/

vector.o: vector.c vector.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c vector.c -o ./target/vector.o

vector_test: vector.* target
	$(CC) -DVECTOR_MAIN $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) vector.c -o ./target/vector_test $(LIBS)

circular_buffer.o: circular_buffer.c circular_buffer.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c circular_buffer.c -o ./target/circular_buffer.o

audio.o: audio.c audio.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c audio.c -o ./target/audio.o

os_filter.o: os_filter.c os_filter.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c os_filter.c -o ./target/os_filter.o

circular_buffer_test: circular_buffer.* target
	$(CC) -DBUFFER_MAIN $(WFLAGS) $(DEBUG_FLAGS) $(ARCH_FLAGS) circular_buffer.c -o ./target/circular_buffer_test $(LIBS)

audio_test: audio.* target
	$(CC) -DAUDIO_MAIN $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) audio.c vector.c circular_buffer.c os_filter.c -o ./target/audio_test $(LIBS)

audio_test_debug: audio.* target
	$(CC) -DAUDIO_MAIN $(WFLAGS) $(DEBUG_FLAGS) $(ARCH_FLAGS) audio.c vector.c circular_buffer.c os_filter.c -o ./target/audio_test_debug $(LIBS)


osfilter_test_debug: os_filter.* target
	$(CC) -DOSFILTER_MAIN $(WFLAGS) $(DEBUG_FLAGS) $(ARCH_FLAGS) os_filter.c vector.c circular_buffer.c -o ./target/osfilter_test_debug $(LIBS)


libfilter: vector.o circular_buffer.o audio.o os_filter.o
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -shared ./target/*.o -o ./target/libfilter.so
