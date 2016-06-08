#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>
#include <string.h>

#include "circular_buffer.h"

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

static inline int __index(CircularBuffer * circular_buffer, long index)
{
    return index < circular_buffer->capacity ?
        (int)index : (int)(index % circular_buffer->capacity);
}


bool CircularBuffer_produce(CircularBuffer * buf, const NUMERIC * data, int length)
{
    pthread_mutex_lock(&buf->offset_mutex);
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

    pthread_mutex_lock(&buf->offset_mutex);
    buf->offset_producer += length;
    pthread_mutex_unlock(&buf->offset_mutex);
    return true;
}

void CircularBuffer_produce_blocking(CircularBuffer * buf, const NUMERIC * data, int length)
{
    pthread_mutex_lock(&buf->offset_mutex);
    int check_counter = 0;
    while (buf->offset_consumer + length + buf->capacity <= buf->offset_producer) {
        pthread_cond_wait(&buf->offset_condition, &buf->offset_mutex);
        if (++check_counter > BLOCK_ABORT) {
            exit(-1);
        }
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

    pthread_mutex_lock(&buf->offset_mutex);
    buf->offset_producer += length;
    pthread_cond_signal(&buf->offset_condition);
    pthread_mutex_unlock(&buf->offset_mutex);
}

bool CircularBuffer_consume(CircularBuffer * buf, NUMERIC * target, int length, int preamble)
{
    pthread_mutex_lock(&buf->offset_mutex);
    if (buf->offset_producer < buf->offset_consumer + length) {
        pthread_mutex_unlock(&buf->offset_mutex);
        return false;
    }
    unsigned long offset = buf->offset_consumer;
    pthread_mutex_unlock(&buf->offset_mutex);

    for (int i = 0; i < length + preamble; ++i) {
        target[i] = buf->data[__index(buf, i + offset - preamble)];
    }
    pthread_mutex_lock(&buf->offset_mutex);
    buf->offset_consumer += length;
    pthread_mutex_unlock(&buf->offset_mutex);
    return true;
}

void CircularBuffer_consume_blocking(CircularBuffer * buf, NUMERIC * target, int length, int preamble)
{
    pthread_mutex_lock(&buf->offset_mutex);
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
    pthread_mutex_lock(&buf->offset_mutex);
    buf->offset_consumer += length;
    pthread_cond_signal(&buf->offset_condition);
    pthread_mutex_unlock(&buf->offset_mutex);
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
