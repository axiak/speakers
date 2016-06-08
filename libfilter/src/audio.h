#pragma once

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
} AudioOptions;


int run_filter(AudioOptions audioOptions);