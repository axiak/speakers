#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <portaudio.h>
#include <sndfile.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#include "audio.h"
#include "common.h"
#include "decoding.h"
#include "circular_buffer.h"
#include "os_filter.h"

#ifdef LINUX_ALSA
#include <pa_linux_alsa.h>
#endif

#define DEBUG_PRINT_INTERVAL_MILLIS (1000)
#define WAV_BUFFER_SIZE (32768)
#define INPUT_FORMAT_CHECK_INTERVAL (2000)


CircularBuffer * __read_audio_file(AudioOptions audio_options);


long currentTimeMillis(void);

typedef struct {
    CircularBuffer * buffer;
    volatile long start_time;
    volatile unsigned long num_frames;
    int number_of_channels;
    int enabled_channels;
    int stripe_input;
} RecordCallbackMetadata;


typedef struct {
    CircularBuffer * buffer;
    int num_output_channels;
} PlaybackCallbackData;


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

    RecordCallbackMetadata * meta_data = (RecordCallbackMetadata *)user_data;

    meta_data->num_frames += frames_per_buffer;

    long current_time = currentTimeMillis();
    long delta = current_time - meta_data->start_time;

    if (delta >= INPUT_FORMAT_CHECK_INTERVAL) {
        printf("time delta: %lu, frames_per_buffer: %lu, frame delta: %lu, InputUnderflow: %d, InputOverflow: %d, OutputUnderflow: %d, OutputOverflow: %d, outputtonow: %f, nowtoinput: %f\n",
               delta,
               frames_per_buffer,
               meta_data->num_frames,
               (status_flags & paInputUnderflow) != 0,
               (status_flags & paInputOverflow) != 0,
               (status_flags & paOutputUnderflow) != 0,
               (status_flags & paOutputOverflow) != 0,
               time_info->outputBufferDacTime - time_info->currentTime,
               time_info->currentTime - time_info->inputBufferAdcTime
               );
        meta_data->num_frames = 0;
        meta_data->start_time = current_time;
    }

    CircularBuffer * buffer = meta_data->buffer;
    const NUMERIC * reader = (const NUMERIC*)input_buffer;


    if (meta_data->stripe_input) {
        CircularBuffer_produce_blocking_striped(buffer,
                                                reader,
                                                frames_per_buffer,
                                                meta_data->number_of_channels,
                                                meta_data->enabled_channels);
    } else {
        CircularBuffer_produce_blocking(buffer, reader, frames_per_buffer * meta_data->number_of_channels);
    }

    return paContinue;
}



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

int is_parent_running(pthread_t parent_tid);

int run_filter(AudioOptions audio_options)
{
    PaStreamParameters input_parameters, output_parameters;
    PaStream *input_stream = NULL, *output_stream = NULL;
    PaError err = paNoError;
    CircularBuffer * output_buffer = CircularBuffer_create(audio_options.buffer_size);
    CircularBuffer * input_buffer = NULL;
    CircularBuffer * decoder_input = NULL;
    DecodingBuffer * decoder_buffer = NULL;

    int live_stream = !audio_options.wav_path || !strlen(audio_options.wav_path);

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

    RecordCallbackMetadata record_data = {
        NULL,
        0,
        0,
        audio_options.number_of_channels,
        audio_options.enabled_channels,
        0
    };

    {
        int num_enabled_channels = 0;

        for (int i = 0; i < audio_options.number_of_channels; ++i) {
            if (audio_options.enabled_channels & (1 << i)) {
                num_enabled_channels++;
            }
        }
        if (num_enabled_channels != audio_options.enabled_channels) {
            record_data.stripe_input = 1;
        }
        if (num_enabled_channels == 0) {
            record_data.stripe_input = 0;
        }
    }

    if (record_data.stripe_input) {
        printf("striping record_data: %d channels with %d enabled flag\n",
               record_data.number_of_channels,
               record_data.enabled_channels
               );
    } else {
        printf("NOT striping record_data: %d channels with %d enabled flag\n",
               record_data.number_of_channels,
               record_data.enabled_channels
               );
    }

    if (live_stream) {
        input_buffer = CircularBuffer_create(audio_options.buffer_size);

        if (audio_options.decode_input) {
            decoder_input = CircularBuffer_create(audio_options.buffer_size);
            decoder_buffer = Decoding_new(decoder_input, input_buffer);
            record_data.buffer = decoder_input;
            if (!Decoding_start_ac3(decoder_buffer)) {
                printf("Failed to start ac3 decoder thread.\n");
                goto done;
            }
        } else {
            record_data.buffer = input_buffer;
        }

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
                            &record_data
                            );

        if (err != paNoError) {
            goto done;
        }

        record_data.start_time = currentTimeMillis();


#ifdef LINUX_ALSA
        PaAlsa_EnableRealtimeScheduling(input_stream, 1);
#endif

        if ((err = Pa_StartStream(input_stream)) != paNoError) {
            goto done;
        }
    } else {
        input_buffer = __read_audio_file(audio_options);
    }

    output_parameters.device = audio_options.output_device;
    if (audio_options.output_channels >= 6) {
        output_parameters.channelCount = audio_options.output_channels + 2;
    } else {
        output_parameters.channelCount = audio_options.output_channels;
    }
    output_parameters.sampleFormat = PA_SAMPLE_TYPE;
    output_parameters.suggestedLatency = Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
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

#ifdef LINUX_ALSA
    PaAlsa_EnableRealtimeScheduling(output_stream, 1);
#endif

    if ((err = Pa_StartStream(output_stream)) != paNoError) {
        goto done;
    }

    int output_scale = output_parameters.channelCount / 2;
    int frame_print_interval = DEBUG_PRINT_INTERVAL_MILLIS * audio_options.sample_rate / 1000;
    int current_frame = 0;

    while (true) {
        if ((err = Pa_IsStreamActive(output_stream)) != 1) {
            break;
        }
        if (input_stream && (err = Pa_IsStreamActive(input_stream)) != 1) {
            break;
        }
        current_frame += OSFilter_execute(filter, input_buffer, output_buffer);
        if (audio_options.print_debug && current_frame > frame_print_interval) {
            current_frame -= frame_print_interval;

            if (!is_parent_running(audio_options.parent_thread_ident)) {
                printf("Parent thread is dead. Shutting down now.\n");
                fflush(stdout);
                goto done;
            }

            if (!audio_options.print_debug) {
                continue;
            }
            long frame_difference = (CircularBuffer_lag(input_buffer) + CircularBuffer_lag(output_buffer) / output_scale) / 2;
            float lag = (float)(frame_difference) / audio_options.sample_rate * 1000;
            printf("%lu\t%lu\t%lu\t%fms\n",
                   input_buffer->offset_producer,
                   output_buffer->offset_consumer / output_scale,
                   frame_difference, lag
                   );

            if (live_stream && (lag > audio_options.lag_reset_limit * 1000)) {
                printf("Resetting to latest due to high lag.\n");
                CircularBuffer_fastforward(input_buffer, audio_options.filter_size * 2);
                CircularBuffer_fastforward(output_buffer, 0);
            }
            fflush(stdout);
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
    if (input_buffer) {
        CircularBuffer_destroy(input_buffer);
    }
    CircularBuffer_destroy(output_buffer);
    if (decoder_input) {
        CircularBuffer_destroy(decoder_input);
    }
    if (decoder_buffer) {
        Decoding_free(decoder_buffer);
    }

    OSFilter_destroy(filter);

    if (err != paNoError) {
        fprintf(stdout, "An error occured while using the portaudio stream\n");
        fprintf(stdout, "Error number: %d\n", err);
        fprintf(stdout, "Error message: %s\n", Pa_GetErrorText(err));
        fflush(stdout);
        err = 1;
    }
    return err;
}

int is_parent_running(pthread_t parent_tid)
{
    return pthread_kill(parent_tid, 0) != ESRCH;
}


CircularBuffer * __read_audio_file(AudioOptions audio_options)
{
    struct stat st;
    SF_INFO sf_info;
    long wav_file_size;

    float * buffer = NULL;
    CircularBuffer * input_buffer = NULL;
    SNDFILE * wav_file = NULL;

    if (stat(audio_options.wav_path, &st) == 0) {
        wav_file_size = st.st_size;
    } else {
        printf("Could not open wav file: %s\n", audio_options.wav_path);
        fflush(stdout);
        goto error;
    }

    input_buffer = CircularBuffer_create(wav_file_size);

    printf("Opening wave file %s with size %ld\n", audio_options.wav_path, wav_file_size);

    if (!(wav_file = sf_open(audio_options.wav_path, SFM_READ, &sf_info))) {
        printf("Could not open wav file: %s\n", audio_options.wav_path);
        fflush(stdout);
        goto error;
    }

    buffer =  (float *)malloc(sizeof(float) * WAV_BUFFER_SIZE);

    if (!buffer) {
        sf_close(wav_file);
        goto error;
    }

    int readcount;
    while ((readcount = sf_read_float(wav_file, buffer, WAV_BUFFER_SIZE))) {
        CircularBuffer_produce_blocking(input_buffer, buffer, readcount);
    }

    free(buffer);
    sf_close(wav_file);
    return input_buffer;

 error:
    if (buffer) {
        free(buffer);
    }
    if (input_buffer) {
        CircularBuffer_destroy(input_buffer);
    }
    if (wav_file) {
        sf_close(wav_file);
    }
    return NULL;
}

long currentTimeMillis(void) {
    struct timespec tspec;
    clock_gettime(CLOCK_REALTIME, &tspec);
    long sec = (int) tspec.tv_sec;
    int msec = (int) ((double) tspec.tv_nsec) / 1000000.0;
    return sec * 1000 + msec;
}

#ifdef AUDIO_MAIN

int main(int argc, char ** argv)
{
    (void)argc;
    (void)argv;


    AudioOptions audio_options = {
        48000,
        0,
        0,
        2,
        NULL,
        1,
        3,
        4,
        1024 * 1024,
        1,
        0.5,
        NULL,
        0.10,
        pthread_self()
    };

    audio_options.filters = (NUMERIC *)malloc(sizeof(NUMERIC) * audio_options.num_filters * audio_options.filter_size);
    for (int i = 0; i < audio_options.num_filters * audio_options.filter_size; ++i) {
        audio_options.filters[i] = 1 / ((double) audio_options.filter_size);
    }

    int ret_val = run_filter(audio_options);

    free(audio_options.filters);

    return ret_val;
}
#endif
