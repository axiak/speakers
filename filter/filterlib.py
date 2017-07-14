import os
import time
import thread
import threading
import numpy as np
from ctypes.util import find_library

from ctypes import CDLL, RTLD_GLOBAL

from cffi import FFI

__all__ = ('run_filter',)

ffi = FFI()


ffi.cdef("""
typedef struct {
    int sample_rate;
    int input_device;
    int output_device;
    int output_channels;
    float * filters;
    int num_filters;
    int filter_size;
    int conv_multiple;
    int buffer_size;
    int print_debug;
    float input_scale;
    const char * wav_file;
    float lag_reset_limit;
    long parent_thread_ident;
    int number_of_channels;
    int enabled_channels;
} AudioOptions;

int run_filter(AudioOptions audioOptions);
""")

CDLL(find_library("portaudio"), mode=RTLD_GLOBAL)
CDLL(find_library("fftw3f"), mode=RTLD_GLOBAL)
CDLL(find_library("sndfile"), mode=RTLD_GLOBAL)

C = ffi.dlopen(
    os.path.join(os.path.dirname(__file__), '..', 'libfilter', 'target', 'libfilter.so'),
    RTLD_GLOBAL
)


def run_filter(options):
    wait_if_necessary()

    filters = options.get('filters', ())
    assert filters, "Required a filters parameter."
    assert all(len(filter_) == len(filters[0])
               for filter_ in filters), "All filters must be the same size."

    filters_expanded = np.concatenate([
        np.array(filter_, dtype=np.float32)
        for filter_ in filters
    ])


    def actually_run():
        wav_file = ffi.new('char[]', options.get('wav_file', ''))

        C.run_filter((
            options.get('sample_rate', 48000),
            options.get('input_device', 2),
            options.get('output_device', options.get('input_device', 2)),
            options.get('output_channels', len(filters) * 2),
            ffi.cast('float *', filters_expanded.ctypes.data),
            len(filters),
            len(filters[0]),
            options.get('conv_multiple', 4),
            options.get('buffer_size', 50 * (1 << 10)),
            int(bool(options.get('print_debug'))),
            float(options.get('input_scale', 0.707)),
            wav_file,
            float(options.get('lag_reset_limit', 0.10)),
            thread.get_ident(),
            int(options.get('input_channels', 2)),
            compute_enabled_channels(options.get('enabled_channels', ()))
        ))

    t = threading.Thread(target=actually_run)

    t.daemon = True
    t.start()
    while t.is_alive():
        t.join(0.10)


def compute_enabled_channels(enabled_channels):
    result = 0
    for channel in enabled_channels:
        result |= (1 << channel)
    return result


def wait_if_necessary():
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        if uptime_seconds < 45:
            print "Waiting 10 seconds to give ALSA a chance."
            time.sleep(10)
    except:
        pass
