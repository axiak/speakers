import sys
import numpy
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
    cutoff_freq = 50
    cutoff_freq_2 = 1600
    sample_freq = 48000

    filters = [
        scipy.signal.remez(
            filter_size,
            [0, cutoff_freq * 0.9, cutoff_freq * 1.1, sample_freq / 2],
            [1, 1],
            Hz=sample_freq,
            maxiter=100,
            grid_density=32
        ),
    ]

    #plot_filter(filters[0], sample_freq)

    from filterlib import run_filter

    run_filter({
        'filters': filters,
        'sample_rate': sample_freq,
        'input_device': 2,
        'output_device': 2
    })



def plot_filter(filter_coefs, sample_freq):
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
    ax2 = ax1.twinx()
    pyplot.semilogx(freq_hz, 180 / numpy.pi * numpy.unwrap(numpy.angle(ampl)), 'r-')

    pyplot.legend(['Amplitude', 'Phase'], loc='upper right')

    pyplot.xlabel('Frequency (Hz)')
    pyplot.ylabel('Phase angle (deg)')
    pyplot.title('Filter frequency response')

    print 'Please close figure window to continue'
    pyplot.show()


if __name__ == '__main__':
    main()

