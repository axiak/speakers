#pragma once

#include <pthread.h>

#include "common.h"

typedef struct {
    int sample_rate;
    int input_device;
    int output_device;
    int output_channels;
    NUMERIC * filters;
    int num_filters;
    int filter_size;
    int conv_multiple;
    int buffer_size;
    int print_debug;
    NUMERIC input_scale;
    const char * wav_path;
    float lag_reset_limit;
    pthread_t parent_thread_ident;
    int number_of_channels;
    int enabled_channels;
} AudioOptions;


int run_filter(AudioOptions audioOptions);
