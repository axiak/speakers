#pragma once

#include <pthread.h>
#include <stdbool.h>

#include "common.h"

typedef struct {
    long offset_producer;
    long offset_consumer;
    int capacity;
    NUMERIC * restrict data;
    pthread_mutex_t offset_mutex;
    pthread_cond_t offset_condition;
} CircularBuffer;

#define BLOCK_ABORT 10

CircularBuffer * CircularBuffer_create(int capacity);
void CircularBuffer_destroy(CircularBuffer * circular_buffer);

bool CircularBuffer_produce(CircularBuffer * circular_buffer, const NUMERIC * data, int length);
void CircularBuffer_produce_blocking(CircularBuffer * circular_buffer, const NUMERIC * data, int length);

bool CircularBuffer_consume(CircularBuffer * circular_buffer, NUMERIC * target, int length, int preamble);
void CircularBuffer_consume_blocking(CircularBuffer * circular_buffer, NUMERIC * target, int length, int preamble);

void CircularBuffer_fastforward(CircularBuffer * circular_buffer, int distance_from_end);
long CircularBuffer_lag(CircularBuffer * circular_buffer);
