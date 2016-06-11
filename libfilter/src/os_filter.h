#pragma once

#include "common.h"
#include "vector.h"
#include "circular_buffer.h"

/* data steps:
   - striped_input
   - scratch_invector (input vector)
   - scratch_covector (vector in codomain)
   - scratch_vector   (output vector)
   - striped_output
 */
typedef struct {
    int num_filters;
    int num_channels;
    int conv_length;
    int step_size;
    NUMERIC * striped_input;
    NUMERIC * striped_output;
    Vector * scratch;
    Vector ** cofilters;
    FFTW_PLAN input_plan;
    FFTW_PLAN output_plan;
    NUMERIC input_scale;
} OSFilter;


OSFilter * OSFilter_create(const NUMERIC * filters,
                           int num_filters,
                           int num_channels,
                           int filter_length,
                           int conv_length,
                           float input_scale);

void OSFilter_destroy(OSFilter * filter);

int OSFilter_execute(OSFilter * filter, CircularBuffer * input, CircularBuffer * output);

