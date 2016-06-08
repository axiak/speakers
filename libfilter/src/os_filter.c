#include <stdlib.h>
#include <fftw3.h>
#include <stdbool.h>

#include "vector.h"
#include "os_filter.h"


// private methods
void __OSFilter_copy_input(OSFilter * filter, const NUMERIC * input);
void __OSFilter_init_cofilters(OSFilter * os_filter, const NUMERIC * filters, int filter_length);
bool __OSFilter_create_vectors(Vector *** vector_ref, int num_vectors, int vector_length);


OSFilter * OSFilter_create(const NUMERIC * filters,
                           int num_filters,
                           int num_channels,
                           int filter_length,
                           int conv_length)
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

    if (!(filter->scratch = Vector_create(conv_length))) {
        OSFilter_destroy(filter);
        return NULL;
    }
    if (!(filter->striped_input = (NUMERIC *)malloc(sizeof(NUMERIC) * conv_length * num_channels))) {
        OSFilter_destroy(filter);
        return NULL;
    }
    // TODO - Figure out how to make this step_size length
    if (!(filter->striped_output = (NUMERIC *)malloc(sizeof(NUMERIC) * conv_length * num_filters * num_channels))) {
        OSFilter_destroy(filter);
        return NULL;
    }
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

void OSFilter_execute(OSFilter * filter)
{
    int i;
    float scaling_factor = 1 / (float)(filter->conv_length);

    for (int filter_idx = 0; filter_idx < filter->num_filters; ++filter_idx) {
        for (int channel_idx = 0; channel_idx < filter->num_channels; ++channel_idx) {
            // copy value into scratch
            for (i = 0; i < filter->conv_length; ++i) {
                filter->striped_input[i * filter->num_channels + channel_idx] *= .5;
                filter->scratch->data[i][0] =
                    filter->striped_input[i * filter->num_channels + channel_idx];
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
                filter->striped_output[i * filter->num_channels * filter->num_filters +
                                       filter->num_filters * filter_idx +
                                       channel_idx] =
                    filter->scratch->data[i][0];
                    //filter->striped_input[i * filter->num_channels + channel_idx];
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
