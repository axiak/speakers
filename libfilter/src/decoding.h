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


DecodingBuffer * new_buffer(CircularBuffer * input, CircularBuffer * output);
void free_buffer(DecodingBuffer * buffer);
void stop_decoder(DecodingBuffer * buffer);

pthread_t start_decoder_ac3(DecodingBuffer * decoding_buffer);

