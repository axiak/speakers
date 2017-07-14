#!/usr/bin/env python
import os
import numpy
from muter import unmute
from filters import FilterFactory


def main():
    ff = filter_factory = FilterFactory(sample_freq=48000, filter_size=1025)
    filters = current_crossover(ff)


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
        'input_device': 9, # 1 for analog
        'output_device': 7,
        'input_scale': .17,
        'print_debug': True,
        'input_channels': 2,
        #'enabled_channels': (1, 2),
        #'wav_file': '/home/axiak/Documents/a2002011001-e02.wav'
    })


def old_crossover(ff):
    filter_factory = ff

    #   EQ will compensate for baffle effects
    #   Use the shelving filter for baffle step compensation only outside the EQ band
    meas_lf_cutoff = 300
    baffle_step = ff.apply_freq_limits(ff.shelf(450, -12), 0, meas_lf_cutoff)
    tw_comp = ff.invert_measurement('../measurements/tweeterleft2.txt.gz', (-1e-3, 3.2e-3), (1.2e3, 1.8e4), nocache=True)
    mid_comp = ff.invert_measurement('../measurements/midleft2.txt.gz', (-1e-3, 3.1e-3), (meas_lf_cutoff, 1.4e4), nocache=True)
    wf_comp = ff.invert_measurement('../measurements/wooferleft2.txt.gz', (-1e-3, 2.5e-3), (meas_lf_cutoff, 3e3), nocache=True)

    RT2 = 0.707

    mid_hp = ff.analog_hp2(600, RT2) ** 2
    mid_lp = ff.analog_lp2(3000, RT2) ** 2
    mid_bp = mid_hp * mid_lp

    tw_eq = ff.parametric_eq(1.9e4, 2, -2)
    #mid_eq = ff.parametric_eq(300, 0.707, 1)
    #wf_eq = ff.parametric_eq(65, 1, 3)

    tw_comp *= tw_eq
    #mid_comp *= mid_eq
    #wf_comp *= wf_eq

    crossover = [
        wf_comp * (ff.analog_lp2(600, RT2) ** 2) * baffle_step * ff.gain(4),
        mid_comp * mid_bp * baffle_step * ff.gain(2),
        tw_comp * (ff.analog_hp2(3000, RT2) ** 2) * baffle_step * ff.invert() * ff.gain(4),
    ]
    return crossover


def current_crossover(ff):
    filter_factory = ff

    bass_driver = ff.measurement('../measurements/wooferleft3.txt.gz', (-.001, 0.0031))
    mid_driver = ff.measurement('../measurements/midleft3.txt.gz', (-.001, 0.0040))
    tw_driver = ff.measurement('../measurements/tweeterleft3.txt.gz', (-.001, 0.0040))

    #   EQ will compensate for baffle effects
    #   Use the shelving filter for baffle step compensation only outside the EQ band
    meas_lf_cutoff = 300
    baffle_step = ff.apply_freq_limits(ff.shelf(450, -10), 0, meas_lf_cutoff)

    RT2 = 0.707 ** .5

    mid_hp = ff.analog_hp2(450, RT2) ** 4
    mid_lp = ff.analog_lp2(3000, RT2) ** 4
    mid_bp = mid_hp * mid_lp

    tw_eq = ff.parametric_eq(1.9e4, 2, -2)
    mid_eq = ff.parametric_eq(300, 0.707, 1)
    wf_eq = ff.parametric_eq(200, 1, 4)

    crossover = [
        wf_eq * (ff.analog_lp2(800, RT2) ** 4) * baffle_step * ff.gain(15),
        mid_eq * mid_bp * baffle_step * ff.gain(-5.0),
        tw_eq * (ff.analog_hp2(3000, RT2) ** 4) * baffle_step * ff.invert() * ff.gain(-4),
    ]

    wf_comp, mid_comp, tw_comp = [ff.invert(driver) for driver in (bass_driver, mid_driver, tw_driver)]

    crossover = [
        ff.min_phase(wf_comp * crossover[0]),
        ff.min_phase(mid_comp * crossover[1]),
        ff.min_phase(tw_comp  * crossover[2])
    ]
    return crossover


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
