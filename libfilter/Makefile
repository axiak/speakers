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

all: libfilter

clean:
	rm -rfv target/

target:
	mkdir target/

vector.o: $(SRC)/vector.c $(SRC)/vector.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c $(SRC)/vector.c -o ./target/vector.o

vector_test: $(SRC)/vector.* target
	$(CC) -DVECTOR_MAIN $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) $SRC/vector.c -o ./target/vector_test $(LIBS)

circular_buffer.o: $(SRC)/circular_buffer.c $(SRC)/circular_buffer.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c $(SRC)/circular_buffer.c -o ./target/circular_buffer.o

audio.o: $(SRC)/audio.c $(SRC)/audio.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c $(SRC)/audio.c -o ./target/audio.o

os_filter.o: $(SRC)/os_filter.c $(SRC)/os_filter.h target
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -c $(SRC)/os_filter.c -o ./target/os_filter.o

circular_buffer_test: $(SRC)/circular_buffer.* target
	$(CC) -DBUFFER_MAIN $(WFLAGS) $(DEBUG_FLAGS) $(ARCH_FLAGS) $(SRC)/circular_buffer.c -o ./target/circular_buffer_test $(LIBS)

audio_test: $(SRC)/audio.* target
	$(CC) -DAUDIO_MAIN $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) $(SRC)/audio.c $(SRC)/vector.c $(SRC)/circular_buffer.c $(SRC)/os_filter.c -o ./target/audio_test $(LIBS)

audio_test_debug: $(SRC)/audio.* target
	$(CC) -DAUDIO_MAIN $(WFLAGS) $(DEBUG_FLAGS) $(ARCH_FLAGS) $(SRC)/audio.c $(SRC)/vector.c $(SRC)/circular_buffer.c $(SRC)/os_filter.c -o ./target/audio_test_debug $(LIBS)


osfilter_test_debug: os_filter.* target
	$(CC) -DOSFILTER_MAIN $(WFLAGS) $(DEBUG_FLAGS) $(ARCH_FLAGS) $(SRC)/os_filter.c $(SRC)/vector.c $(SRC)/circular_buffer.c -o ./target/osfilter_test_debug $(LIBS)


libfilter: vector.o circular_buffer.o audio.o os_filter.o
	$(CC) $(WFLAGS) $(OPTFLAGS) $(ARCH_FLAGS) -shared ./target/*.o -o ./target/libfilter.so
