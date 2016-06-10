#include <stdlib.h>
#include <stdio.h>
#include <portaudio.h>
#include <pa_linux_alsa.h>

#include "audio.h"
#include "common.h"
#include "circular_buffer.h"
#include "os_filter.h"

static int recordCallback(const void *input_buffer,
                          void *output_buffer,
                          unsigned long frames_per_buffer,
                          const PaStreamCallbackTimeInfo* time_info,
                          PaStreamCallbackFlags status_flags,
                          void *user_data)
{
    (void)output_buffer;
    (void)time_info;
    (void)status_flags;

    CircularBuffer * buffer = (CircularBuffer *)user_data;
    const NUMERIC * reader = (const NUMERIC*)input_buffer;


    CircularBuffer_produce_blocking(buffer, reader, frames_per_buffer * 2);

    return paContinue;
}


typedef struct {
    CircularBuffer * buffer;
    int num_output_channels;
} PlaybackCallbackData;

static int playCallback(const void *input_buffer,
                        void *output_buffer,
                        unsigned long frames_per_buffer,
                        const PaStreamCallbackTimeInfo* time_info,
                        PaStreamCallbackFlags status_flags,
                        void *user_data)
{
    (void)input_buffer;
    (void)time_info;
    (void)status_flags;

    PlaybackCallbackData * callback_data = (PlaybackCallbackData *)user_data;
    CircularBuffer * buffer = callback_data->buffer;
    NUMERIC * writer = (NUMERIC *)output_buffer;
    CircularBuffer_consume_blocking(buffer, writer, frames_per_buffer * callback_data->num_output_channels, 0);
    return paContinue;
}


int run_filter(AudioOptions audio_options)
{
    PaStreamParameters input_parameters, output_parameters;
    PaStream *input_stream = NULL, *output_stream = NULL;
    PaError err = paNoError;

    CircularBuffer * input_buffer = CircularBuffer_create(audio_options.buffer_size);
    CircularBuffer * output_buffer = CircularBuffer_create(audio_options.buffer_size);

    OSFilter * filter =
        OSFilter_create(
                        audio_options.filters,
                        audio_options.num_filters,
                        2,
                        audio_options.filter_size,
                        audio_options.conv_multiple * (audio_options.filter_size - 1),
                        audio_options.input_scale
                        );
    if (!filter) {
        goto done;
    }

    if ((err = Pa_Initialize()) != paNoError) {
        goto done;
    }

    int step_size = audio_options.filter_size - 1;

    input_parameters.device = audio_options.input_device;
    input_parameters.channelCount = 2;
    input_parameters.sampleFormat = PA_SAMPLE_TYPE;
    input_parameters.suggestedLatency = Pa_GetDeviceInfo(input_parameters.device)->defaultHighInputLatency;
    input_parameters.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream(
                        &input_stream,
                        &input_parameters,
                        NULL,
                        audio_options.sample_rate,
                        step_size,
                        paNoFlag,
                        recordCallback,
                        input_buffer
                        );

    if (err != paNoError) {
        goto done;
    }

    PaAlsa_EnableRealtimeScheduling(input_stream, 1);

    if ((err = Pa_StartStream(input_stream)) != paNoError) {
        goto done;
    }

    output_parameters.device = audio_options.output_device;
    if (audio_options.output_channels >= 6) {
        output_parameters.channelCount = audio_options.output_channels + 2;
    } else {
        output_parameters.channelCount = audio_options.output_channels;
    }
    output_parameters.sampleFormat = PA_SAMPLE_TYPE;
    output_parameters.suggestedLatency = Pa_GetDeviceInfo(output_parameters.device)->defaultHighOutputLatency;
    output_parameters.hostApiSpecificStreamInfo = NULL;

    PlaybackCallbackData playback_data = {
        output_buffer,
        output_parameters.channelCount
    };

    printf("output channels: %d\n", output_parameters.channelCount);

    err = Pa_OpenStream(
                        &output_stream,
                        NULL,
                        &output_parameters,
                        audio_options.sample_rate,
                        step_size,
                        paNoFlag,
                        playCallback,
                        &playback_data
                        );

    if (err != paNoError) {
        goto done;
    }

    PaAlsa_EnableRealtimeScheduling(output_stream, 1);

    if ((err = Pa_StartStream(output_stream)) != paNoError) {
        goto done;
    }

    int output_scale = output_parameters.channelCount / 2;

    while ((err = Pa_IsStreamActive(input_stream)) == 1 &&
           (err = Pa_IsStreamActive(output_stream)) == 1) {
        OSFilter_execute(filter, input_buffer, output_buffer);
        if (audio_options.print_debug) {
            printf("%lu\t%lu\t%lu\n", input_buffer->offset_producer, output_buffer->offset_consumer / output_scale,
                   input_buffer->offset_producer - output_buffer->offset_consumer / output_scale);
        }
    }
    if (err < 0) {
        goto done;
    }

done:
    if (output_stream) {
        Pa_CloseStream(output_stream);
    }
    if (input_stream) {
        Pa_CloseStream(input_stream);
    }
    Pa_Terminate();
    CircularBuffer_destroy(input_buffer);
    CircularBuffer_destroy(output_buffer);
    OSFilter_destroy(filter);

    if (err != paNoError) {
        fprintf(stderr, "An error occured while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        err = 1;
    }
    return err;
}


#ifdef AUDIO_MAIN


int main(int argc, char ** argv)
{
    (void)argc;
    (void)argv;

    AudioOptions audio_options = {
        48000,
        2,
        2,
        6,
        NULL,
        3,
        1025,
        4,
        1024 * 1024,
        1,
        .5
    };

    audio_options.filters = (NUMERIC *)malloc(sizeof(NUMERIC) * 1025 * 3);
    for (int i = 0; i < 1025 * 3; ++i) {
        audio_options.filters[i] = 1 / 14096.0;
    }

    int ret_val = run_filter(audio_options);

    free(audio_options.filters);

    return ret_val;
}


#endif
