#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>
#include <string.h>
#include <errno.h>

#include "circular_buffer.h"

static inline int __index(CircularBuffer * circular_buffer, long index);
static inline int __mod(long a, long b);
static inline void plock(pthread_mutex_t *mutex);


CircularBuffer * CircularBuffer_create(int capacity)
{
    CircularBuffer * buffer = (CircularBuffer *)malloc(sizeof(CircularBuffer));
    if (!buffer) {
        return NULL;
    }

    FFTW_COMPLEX * data = (FFTW_COMPLEX*)FFTW_MALLOC(sizeof(FFTW_COMPLEX) * ((capacity / 2) + 1));
    if (!data) {
        CircularBuffer_destroy(buffer);
        return NULL;
    }

    buffer->data = (NUMERIC *)data;

    pthread_mutex_init(&buffer->offset_mutex, NULL);
    pthread_cond_init(&buffer->offset_condition, NULL);

    buffer->offset_producer = 0;
    buffer->offset_consumer = 0;

    buffer->capacity = capacity;
    return buffer;
}

void CircularBuffer_destroy(CircularBuffer * circular_buffer)
{
    if (circular_buffer) {
        if (circular_buffer->data) {
            FFTW_FREE(circular_buffer->data);
        }
        pthread_mutex_destroy(&circular_buffer->offset_mutex);
        pthread_cond_destroy(&circular_buffer->offset_condition);
        free(circular_buffer);
    }
}


void CircularBuffer_produce_blocking_striped(CircularBuffer * buf,
                                             const NUMERIC * data,
                                             int number_of_frames,
                                             int stripe_length,
                                             int enabled_channels_flags)
{
    int enabled_stripes = 0;
    for (int i = 0; i < stripe_length; ++i) {
        if (enabled_channels_flags & (1 << i)) {
            ++enabled_stripes;
        }
    }

    int effective_length = number_of_frames * enabled_stripes;

    int start_index = __index(buf, buf->offset_producer);
    pthread_mutex_unlock(&buf->offset_mutex);

    for (int frame = 0; frame < number_of_frames; ++frame) {
        for (int channel = 0; channel < stripe_length; ++channel) {
            if (enabled_channels_flags & (1 << channel)) {
                buf->data[__index(buf, start_index)] = data[frame + channel];
                ++start_index;
            }
        }
    }
    plock(&buf->offset_mutex);
    buf->offset_producer += effective_length;
    pthread_cond_signal(&buf->offset_condition);
    pthread_mutex_unlock(&buf->offset_mutex);
}


bool CircularBuffer_produce(CircularBuffer * buf, const NUMERIC * data, int length)
{
    plock(&buf->offset_mutex);
    if (buf->offset_consumer + length + buf->capacity <= buf->offset_producer) {
        pthread_mutex_unlock(&buf->offset_mutex);
        return false;
    }
    int start_index = __index(buf, buf->offset_producer);
    pthread_mutex_unlock(&buf->offset_mutex);

    // fast lane
    if (start_index + length < buf->capacity) {
        memcpy(buf->data + start_index, data, sizeof(NUMERIC) * length);
    } else {
        int end_index = start_index + length;
        for (int j = 0; start_index < end_index; ++start_index, ++j) {
            buf->data[__index(buf, start_index)] = data[j];
        }
    }

    plock(&buf->offset_mutex);
    buf->offset_producer += length;
    pthread_mutex_unlock(&buf->offset_mutex);
    return true;
}

void CircularBuffer_produce_blocking(CircularBuffer * buf, const NUMERIC * data, int length)
{
    int start_index = __index(buf, buf->offset_producer);

    // fast lane
    if (start_index + length < buf->capacity) {
        memcpy(buf->data + start_index, data, sizeof(NUMERIC) * length);
    } else {
        int end_index = start_index + length;
        for (int j = 0; start_index < end_index; ++start_index, ++j) {
            buf->data[__index(buf, start_index)] = data[j];
        }
    }

    plock(&buf->offset_mutex);
    buf->offset_producer += length;
    pthread_cond_signal(&buf->offset_condition);
    pthread_mutex_unlock(&buf->offset_mutex);
}

bool CircularBuffer_consume(CircularBuffer * buf, NUMERIC * target, int length, int preamble)
{
    plock(&buf->offset_mutex);
    if (buf->offset_producer < buf->offset_consumer + length) {
        pthread_mutex_unlock(&buf->offset_mutex);
        return false;
    }
    unsigned long offset = buf->offset_consumer;
    pthread_mutex_unlock(&buf->offset_mutex);

    for (int i = 0; i < length + preamble; ++i) {
        target[i] = buf->data[__index(buf, i + offset - preamble)];
    }
    plock(&buf->offset_mutex);
    buf->offset_consumer += length;
    pthread_mutex_unlock(&buf->offset_mutex);
    return true;
}

void CircularBuffer_consume_blocking(CircularBuffer * buf, NUMERIC * target, int length, int preamble)
{
    plock(&buf->offset_mutex);
    int check_counter = 0;
    while (buf->offset_producer < buf->offset_consumer + length) {
        pthread_cond_wait(&buf->offset_condition, &buf->offset_mutex);
        if (++check_counter > BLOCK_ABORT) {
            exit(-1);
        }
    }
    unsigned long offset = buf->offset_consumer;
    pthread_mutex_unlock(&buf->offset_mutex);

    for (int i = 0; i < length + preamble; ++i) {
        target[i] = buf->data[__index(buf, i + offset - preamble)];
    }
    plock(&buf->offset_mutex);
    buf->offset_consumer += length;
    pthread_cond_signal(&buf->offset_condition);
    pthread_mutex_unlock(&buf->offset_mutex);
}


long CircularBuffer_lag(CircularBuffer * buf)
{
    plock(&buf->offset_mutex);
    long result = buf->offset_producer - buf->offset_consumer;
    pthread_mutex_unlock(&buf->offset_mutex);
    return result;
}

void CircularBuffer_fastforward(CircularBuffer * buf, int distance_from_end)
{
    plock(&buf->offset_mutex);

    buf->offset_consumer = buf->offset_producer - distance_from_end;

    pthread_cond_signal(&buf->offset_condition);
    pthread_mutex_unlock(&buf->offset_mutex);
}


static inline int __index(CircularBuffer * circular_buffer, long index)
{
    return __mod(index, circular_buffer->capacity);
}

static inline int __mod(long a, long b)
{
    if (a < 0) {
        if (b < 0) {
            return __mod(-a, -b);
        }
        while (a < 0) {
            a += b;
        }
    }
    return a % b;
}

static inline void plock(pthread_mutex_t *mutex)
{
    int result;
    while (true) {
        result = pthread_mutex_lock(mutex);
        if (result == 0) {
            return;
        } else if (result == EBUSY) {
            continue;
        } else {
            fprintf(stderr, "Error locking thread: %d\n", result);
            fflush(stderr);
            exit(1);
        }
    }
}

#ifdef BUFFER_MAIN

int main(int argc, char ** argv)
{
    CircularBuffer * buf = CircularBuffer_create(1024);
    float arr[] = {1, 2, 3, 4, 5, 6};
    float bar[] = {9, 10};

    printf("%d\n", CircularBuffer_produce(buf, arr, 6));
    printf("%d\n", CircularBuffer_produce(buf, bar, 2));

    for (int i = 0; i < 8; ++i) {
        printf("%f\n", buf->data[i]);
    }

    CircularBuffer_consume(buf, bar, 2);

    for (int i = 0 ; i < 2; ++i) {
        printf("%f\n", bar[i]);
    }

    printf("%lu %lu\n", buf->offset_consumer, buf->offset_producer);
    CircularBuffer_destroy(buf);
    return 0;
}

#endif
