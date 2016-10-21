#!/usr/bin/env python
import os
import numpy
from muter import unmute
from filters import FilterFactory


def main():
    ff = filter_factory = FilterFactory(sample_freq=48000, filter_size=1025)

    baffle_step = ff.shelf(350, -4.5)

    tw_comp = ff.invert_measurement('./measurements/tweeterright.txt.gz', (-1e-3, 4e-3), (1.2e3, 1.8e4))
    mid_comp = ff.invert_measurement('./measurements/midright.txt.gz', (-1e-3, 4e-3), (400, 1e4))
    wf_comp = ff.invert_measurement('./measurements/bassright.txt.gz', (-1e-3, 3.1e-3), (400, 3e3))

    RT2 = 0.707

    mid_hp = ff.analog_hp2(550, RT2) ** 2
    mid_lp = ff.analog_lp2(4500, RT2) ** 2
    mid_bp = mid_hp * mid_lp

    crossover = [
        wf_comp * (ff.analog_lp2(450, RT2) ** 2) * baffle_step,
        mid_comp * mid_bp * baffle_step * ff.gain(4),
        tw_comp * (ff.analog_hp2(2700, RT2) ** 2) * ff.gain(-2.5) * ff.invert()
    ]

    filters = crossover

    try:
        for filter_ in filters:
            filter_.plot()

        reduce(lambda a, b: a + b, filters).plot()
    except:
        pass

    from filterlib import run_filter

    print "Running with filters: {0}".format(filters)

    unmute()

    run_filter({
        'filters': [numpy.real(f.coefficients) for f in filters],
        'sample_rate': filter_factory.sample_freq,
        'input_device': 1, # 9 for spdif
        'output_device': 7,
        'input_scale': 0.4,
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
