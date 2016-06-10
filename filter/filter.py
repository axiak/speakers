import sys
import numpy
import numpy as np
import scipy.fftpack

import scipy.signal
from matplotlib import pyplot


def main():
    """
        scipy.signal.remez(
            filter_size,
            [0, cutoff_freq * 0.9,
             cutoff_freq * 1.1, cutoff_freq_2 * 0.9,
             cutoff_freq_2 * 1.1, sample_freq / 2],
            [0, 1, 0],
            Hz=sample_freq,
            maxiter=50
        ),
        scipy.signal.remez(
            filter_size,
            [0, cutoff_freq_2 * 0.9, cutoff_freq_2 * 1.1, sample_freq / 2],
            [0, 1],
            Hz=sample_freq,
            maxiter=50
        ),
        """

    filter_size = 1025
    cutoff_freq = 300
    cutoff_freq_2 = 2500
    sample_freq = 44100

    transition_width = 125

    print "Building filters..."

    filters = [
        scipy.signal.remez(
            filter_size,
            [0, cutoff_freq - transition_width,
             cutoff_freq + transition_width, sample_freq / 2],
            [1, 0],
            Hz=sample_freq,
            maxiter=100,
            grid_density=64
        ),
        scipy.signal.remez(
            filter_size,
            [0, cutoff_freq - transition_width,
             cutoff_freq + transition_width, cutoff_freq_2 - transition_width,
             cutoff_freq_2 + transition_width, sample_freq / 2],
            [0, 1, 0],
            Hz=sample_freq,
            maxiter=100,
            grid_density=64
        ),
        scipy.signal.remez(
            filter_size,
            [0, cutoff_freq_2 - transition_width,
             cutoff_freq_2 + transition_width, sample_freq / 2],
            [0, 1],
            Hz=sample_freq,
            maxiter=50
        ),
    ]

    #filters = filters[:2]

    #for filter_ in filters:
    #    filter_[0] = filter_[-1] = 0

    try:
        for filter_ in filters:
            plot_filter(filter_, sample_freq)
    except:
        pass

    from filterlib import run_filter

    run_filter({
        'filters': filters,
        'sample_rate': sample_freq,
        'input_device': 2,
        'output_device': 9,
        'print_debug': True
    })


def plot_filter(filter_coefs, sample_freq):

    #fig = pyplot.figure()
    #ax1 = fig.add_subplot(111)
    #freq, response = scipy.signal.freqz(filter_coefs)

    #ax1.semilogy(freq / (2 * np.pi), np.abs(response), 'b-')
    #pyplot.show()
    #return
    t = numpy.linspace(0, float(len(filter_coefs - 1)) / sample_freq, len(filter_coefs))
    fig = pyplot.figure(figsize=(10, 10))

    pyplot.subplot(211)
    pyplot.plot(t, filter_coefs)
    pyplot.grid(True)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Filter coefficient')
    pyplot.title('Filter impulse response')

    ax1 = fig.add_subplot(212)
    (freq, ampl) = scipy.signal.freqz(filter_coefs)
    freq_hz = freq * sample_freq / (2 * numpy.pi)
    pyplot.semilogx(freq_hz, 20 * numpy.log10(numpy.abs(ampl)), 'b-')
    pyplot.grid(True)
    pyplot.ylabel('Amplitude (dB)')
    pyplot.legend(['Amplitude', 'Phase'], loc='upper right')

    ax1.twinx()
    pyplot.semilogx(freq_hz, 180 / numpy.pi * numpy.unwrap(numpy.angle(ampl)), 'r-')
    pyplot.ylabel('Phase angle (deg)')
    pyplot.xlabel('Frequency (Hz)')
    pyplot.legend(['Phase'], loc='lower right')
    pyplot.title('Filter frequency response')

    print 'Please close figure window to continue'
    pyplot.show()


if __name__ == '__main__':
    main()

