#pragma once

#include "circular_buffer.h"

#include <pthread.h>

typedef struct {
    CircularBuffer * input;
    CircularBuffer * output;
    int stop;
    pthread_mutex_t stop_mutex;
    pthread_t thread_id;
} DecodingBuffer;


DecodingBuffer * Decoding_new(CircularBuffer * input, CircularBuffer * output);
void Decoding_free(DecodingBuffer * buffer);
void Decoding_stop(DecodingBuffer * buffer);

pthread_t Decoding_start_ac3(DecodingBuffer * decoding_buffer);

