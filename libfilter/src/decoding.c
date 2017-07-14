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
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>

#include "decoding.h"
#include "common.h"
#include "circular_buffer.h"


#define INBUF_SIZE 4096
#define AUDIO_INBUF_SIZE 20480
#define AUDIO_REFILL_THRESH 4096

int is_stopped(DecodingBuffer * buffer);


void * DecoderThread(void * buffer_void)
{
    DecodingBuffer * buffer = (DecodingBuffer *)buffer_void;

    AVCodec *codec;
    AVCodecContext *c = NULL;
    int len;
    uint8_t inbuf[AUDIO_INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];
    AVPacket avpkt;
    AVFrame *decoded_frame = NULL;

    av_init_packet(&avpkt);

    codec = avcodec_find_decoder(AV_CODEC_ID_AC3);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);

    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    avpkt.data = inbuf;

    int size_to_read = sizeof(inbuf) / sizeof(NUMERIC);

    while (!is_stopped(buffer)) {
        CircularBuffer_consume_blocking(buffer->input, inbuf, size_to_read, 0);
        avpkt.size = size_to_read;

        int got_frame = 0;

        if (!decoded_frame) {
            if (!(decoded_frame = avcodec_alloc_frame())) {
                fprintf(stderr, "out of memory\n");
                exit(1);
            }
        } else {
            avcodec_get_frame_defaults(decoded_frame);
        }

        len = avcodec_decode_audio4(c, decoded_frame, &got_frame, &avpkt);
        if (len < 0) {
            fprintf(stderr, "Error while decoding\n");
            exit(1);
        }
        if (got_frame) {
            /* if a frame has been decoded, output it */
            int data_size = av_samples_get_buffer_size(NULL, c->channels,
                                                       decoded_frame->nb_samples,
                                                       c->sample_fmt, 1);
            CircularBuffer_produce_blocking(buffer->output, decoded_frame->data, data_size / sizeof(NUMERIC));
        }
        avpkt.size -= len;
        avpkt.data += len;
        avpkt.dts = avpkt.pts = AV_NOPTS_VALUE;
        /*
        if (avpkt.size < AUDIO_REFILL_THRESH) {
            / * Refill the input buffer, to avoid trying to decode
             * incomplete frames. Instead of this, one could also use
             * a parser, or use a proper container format through
             * libavformat. * /
            memmove(inbuf, avpkt.data, avpkt.size);
            avpkt.data = inbuf;
            if (len > 0)
                avpkt.size += len;
                }
    */

    }

    avcodec_close(c);
    av_free(c);
    av_free(decoded_frame);

    pthread_exit(NULL);
}


pthread_t start_decoder_ac3(DecodingBuffer * decoding_buffer)
{
    int rc;
    if ((rc = pthread_create(&decoding_buffer->thread_id, NULL, DecoderThread, (void *)decoding_buffer))) {
        printf("ERROR: failed to spawn decoder thread: %d\n", rc);
        return NULL;
    }

    return decoding_buffer->thread_id;
}




DecodingBuffer * new_buffer(CircularBuffer * input, CircularBuffer * output)
{
    DecodingBuffer * buffer = (DecodingBuffer *)malloc(sizeof(DecodingBuffer));
    if (buffer == NULL) {
        return NULL;
    }

    buffer->input = input;
    buffer->output = output;
    buffer->stop = 0;
    return buffer;
}


void free_buffer(DecodingBuffer * buffer)
{
    if (buffer != NULL) {
        free(buffer);
    }
}


int is_stopped(DecodingBuffer * buffer)
{
    int ret = 0;
    pthread_mutex_lock(&buffer->stop_mutex);
    ret = buffer->stop;
    pthread_mutex_unlock(&buffer->stop_mutex);
    return ret;
}


void stop_decoder(DecodingBuffer * buffer)
{
    pthread_mutex_lock(&buffer->stop_mutex);
    buffer->stop = 1;
    pthread_mutex_unlock(&buffer->stop_mutex);
    pthread_cancel(buffer->thread_id);
}



