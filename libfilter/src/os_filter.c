#include <stdlib.h>
#include <fftw3.h>
#include <stdbool.h>
#include <string.h>

#include "vector.h"
#include "os_filter.h"


// We will skip 2 in the output because
// 5.1 surround sound doesn't have 6 channels
// so we have to skip channels 4-5.
#define SKIP_FILTER_NUM (2)


// private methods
static int __OSFilter_effective_num_outputs(OSFilter * filter);
void __OSFilter_copy_input(OSFilter * filter, const NUMERIC * input);
void __OSFilter_init_cofilters(OSFilter * os_filter, const NUMERIC * filters, int filter_length);
bool __OSFilter_create_vectors(Vector *** vector_ref, int num_vectors, int vector_length);
void __OSFilter_evaluate(OSFilter * filter);

OSFilter * OSFilter_create(const NUMERIC * filters,
                           int num_filters,
                           int num_channels,
                           int filter_length,
                           int conv_length,
                           float input_scale)
{
    OSFilter * filter = (OSFilter *)malloc(sizeof(OSFilter));
    if (!filter) {
        return NULL;
    }
    filter->cofilters = NULL;
    filter->striped_input = NULL;
    filter->striped_output = NULL;
    filter->scratch = NULL;
    filter->num_filters = num_filters;
    filter->num_channels = num_channels;
    filter->conv_length = conv_length;
    filter->step_size = filter_length - 1;
    filter->input_scale = input_scale;

    if (!(filter->scratch = Vector_create(conv_length))) {
        OSFilter_destroy(filter);
        return NULL;
    }
    if (!(filter->striped_input = (NUMERIC *)malloc(sizeof(NUMERIC) * conv_length * num_channels))) {
        OSFilter_destroy(filter);
        return NULL;
    }

    int effective_filter_size = __OSFilter_effective_num_outputs(filter);

    // TODO - Figure out how to make this step_size length
    if (!(filter->striped_output = (NUMERIC *)malloc(sizeof(NUMERIC) * conv_length * effective_filter_size  * num_channels))) {
        OSFilter_destroy(filter);
        return NULL;
    }
    memset(
           filter->striped_output,
           0,
           sizeof(NUMERIC) * conv_length * effective_filter_size * num_channels
           );


    if (!(__OSFilter_create_vectors(&filter->cofilters, num_filters, conv_length))) {
        OSFilter_destroy(filter);
        return NULL;
    }

    Vector_zero(filter->scratch);

    __OSFilter_init_cofilters(filter, filters, filter_length);

    filter->input_plan = fftwf_plan_dft_1d(
                                           conv_length,
                                           filter->scratch->data,
                                           filter->scratch->data,
                                           FFTW_FORWARD,
                                           FFTW_MEASURE
                                           );
    filter->output_plan = fftwf_plan_dft_1d(
                                           conv_length,
                                           filter->scratch->data,
                                           filter->scratch->data,
                                           FFTW_BACKWARD,
                                           FFTW_MEASURE
                                           );
    return filter;
}



void OSFilter_destroy(OSFilter * filter)
{
    if (filter) {
        if (filter->striped_input) {
            free(filter->striped_input);
            filter->striped_input = NULL;
        }
        if (filter->striped_output) {
            free(filter->striped_output);
            filter->striped_output = NULL;
        }
        if (filter->cofilters) {
            for (int i = 0; i < filter->num_filters; ++i) {
                Vector_destroy(filter->cofilters[i]);
            }
            free(filter->cofilters);
            filter->cofilters = NULL;
        }
        Vector_destroy(filter->scratch);

        fftwf_destroy_plan(filter->input_plan);
        fftwf_destroy_plan(filter->output_plan);
        free(filter);
    }
}




int OSFilter_execute(OSFilter * filter, CircularBuffer * input, CircularBuffer * output)
{
    int step_data_size = (filter->conv_length - filter->step_size) * 2;
    int preamble = filter->step_size * 2;
    int output_scale = __OSFilter_effective_num_outputs(filter);

    CircularBuffer_consume_blocking(
                                    input,
                                    filter->striped_input,
                                    step_data_size,
                                    preamble
                                    );
    __OSFilter_evaluate(filter);
    CircularBuffer_produce_blocking(
                                    output,
                                    filter->striped_output + preamble * output_scale,
                                    step_data_size * output_scale
                                    );
    return step_data_size / 2;
}


void __OSFilter_evaluate(OSFilter * filter)
{
    int i;
    float scaling_factor = 1 / (float)(filter->conv_length);

    int effective_num_filters = __OSFilter_effective_num_outputs(filter);

    for (int filter_idx = 0; filter_idx < filter->num_filters; ++filter_idx) {
        int effective_filter_idx = (filter_idx >= SKIP_FILTER_NUM) ?
            filter_idx + 1 : filter_idx;

        for (int channel_idx = 0; channel_idx < filter->num_channels; ++channel_idx) {
            // copy value into scratch
            for (i = 0; i < filter->conv_length; ++i) {
                filter->scratch->data[i][0] =
                    filter->striped_input[i * filter->num_channels + channel_idx];
                filter->scratch->data[i][0] *= filter->input_scale;
                // since this is a scratch, the imaginary part was probably set
                // to a nonzero number previously
                filter->scratch->data[i][1] = 0;
            }

            // DFT: scratch into freq domain
            fftwf_execute(filter->input_plan);

            // convolution
            Vector_multiply(
                            filter->scratch,
                            filter->scratch,
                            filter->cofilters[filter_idx]
                            );

            // rescale
            for (i = 0; i < filter->conv_length; ++i) {
                filter->scratch->data[i][0] *= scaling_factor;
                filter->scratch->data[i][1] *= scaling_factor;
            }

            // IDFT: freq -> time domain
            fftwf_execute(filter->output_plan);

            // copy value out
            for (i = 0; i < filter->conv_length; ++i) {
                filter->striped_output[i * filter->num_channels * effective_num_filters +
                                       filter->num_channels * effective_filter_idx +
                                       channel_idx] =
                    filter->scratch->data[i][0];
            }
        }
    }
}


void __OSFilter_init_cofilters(OSFilter * os_filter, const NUMERIC * filters, int filter_length)
{

    FFTW_PLAN plan;
    Vector * fft_input = Vector_create(os_filter->conv_length);


    for (int i = 0; i < os_filter->num_filters; ++i) {
        Vector_zero(fft_input);
        Vector_set_real(fft_input, filters, i * filter_length, filter_length);

        plan = fftwf_plan_dft_1d(
                                 os_filter->conv_length,
                                 fft_input->data,
                                 os_filter->cofilters[i]->data,
                                 FFTW_FORWARD,
                                 FFTW_ESTIMATE
                                 );
        fftwf_execute(plan);

        fftwf_destroy_plan(plan);
    }
    Vector_destroy(fft_input);
}

static int __OSFilter_effective_num_outputs(OSFilter * filter)
{
    if (filter->num_filters >= (SKIP_FILTER_NUM + 1)) {
        return filter->num_filters + 1;
    } else {
        return filter->num_filters;
    }
}

bool __OSFilter_create_vectors(Vector *** vector_ref, int num_vectors, int vector_length)
{
    Vector ** vectors = (Vector **)calloc(num_vectors, sizeof(Vector *));
    if (!vectors) {
        return false;
    }
    for (int i = 0; i < num_vectors; ++i) {
        if (!(vectors[i] = Vector_create(vector_length))) {
            return false;
        }
    }
    *vector_ref = vectors;
    return true;
}



#ifdef OSFILTER_MAIN

int main(int argc, char ** argv)
{
    (void)argc;
    (void)argv;

    NUMERIC filter[] = {
        0, 1, 2, 3,
        1, 2, 3, 4
    };
    OSFilter * osfilter = OSFilter_create(filter,
                                          2,
                                          2,
                                          4,
                                          8
                                          );

    OSFilter_destroy(osfilter);
    return 0;
}

#endif
