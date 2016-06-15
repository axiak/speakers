#!/usr/bin/env python
import numpy
from filters import FilterFactory


def main():
    filter_factory = FilterFactory(sample_freq=44100, filter_size=1025)

    filters = [filter_factory.allpass()] * 3

    filters = build_iir_filters(filter_factory)

    filters = build_remez_filters(filter_factory)

    filters[0] = filter_factory.allpass()

    try:
        for filter_ in filters:
            filter_.plot()

        reduce(lambda a, b: a + b, filters).plot()
    except:
        pass

    from filterlib import run_filter

    print "Running with filters: {0}".format(filters)

    run_filter({
        'filters': [numpy.real(f.coefficients) for f in filters],
        'sample_rate': filter_factory.sample_freq,
        'input_device': 9,
        'output_device': 7,
        'print_debug': True,
        #'wav_file': '/home/axiak/Documents/a2002011001-e02.wav'
    })


def build_iir_filters(filter_factory):
    cutoff_freq = 310
    cutoff_freq_2 = 2800

    return [
        filter_factory.butter_filter(
            4,
            cutoff_freq,
            btype='lowpass',
            name='Lowpass at {0}'.format(cutoff_freq)
        ),
        filter_factory.butter_filter(
            4,
            [cutoff_freq, cutoff_freq_2],
            btype='bandpass',
            name="Bandpass for {0}hz-{1}hz".format(cutoff_freq, cutoff_freq_2)
        ),
        filter_factory.butter_filter(
            4,
            cutoff_freq_2,
            btype='highpass',
            name="Highpass for {0}hz".format(cutoff_freq_2)
        )
    ]


def build_remez_filters(filter_factory):
    cutoff_freq = 310
    cutoff_freq_2 = 3000

    transition_width = 125
    transition_width_2 = 100

    return [
        filter_factory.remez_filter(
            [cutoff_freq - transition_width, cutoff_freq + transition_width],
            [1, 0],
            name='Lowpass at {0}hz'.format(cutoff_freq)
        ),
        filter_factory.remez_filter(
            [cutoff_freq - transition_width, cutoff_freq + transition_width,
             cutoff_freq_2 - transition_width_2, cutoff_freq_2 + transition_width_2],
            [0, 1, 0],
            name='Bandpass {0}hz-{1}hz'.format(cutoff_freq, cutoff_freq_2)
        ),
        filter_factory.remez_filter(
            [cutoff_freq_2 - transition_width_2, cutoff_freq_2 + transition_width_2],
            [0, 1],
            name='Highpass at {0}hz'.format(cutoff_freq_2)
        ),
    ]


if __name__ == '__main__':
    main()
