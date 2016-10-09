#!/usr/bin/env python

from filter.filters import FilterFactory, Filter
import scipy.fftpack
import numpy
from matplotlib import pyplot

import pdb

F_s = 48000.
N_filt = 1024
dt = 1.0 / F_s
ff = FilterFactory(F_s, N_filt)

impulse_box = (-1e-3, 4e-3)
freq_box = (1.5e3, 1.8e4)

#   Some utility functions

def dict_to_arrays(h_dict):
    keys = h_dict.keys()
    keys.sort()
    
    t = numpy.array(keys)
    h = numpy.array([h_dict[x] for x in keys])
    
    return (t, h)

def normalize_ir(h):
    return h / ((numpy.sum(h ** 2)) ** 0.5)

def normalize_fr(freq, ampl_db, freq_box):
    #   Make the mean magnitude within freq_box = 0 dB
    inds = (freq > freq_box[0]) * (freq < freq_box[1])
    return ampl_db - numpy.mean(ampl_db[inds])

def ir_to_fr(h, F_s=4.8e4):
    N = h.shape[0]
    H = scipy.fftpack.fft(h)
    
    freq = numpy.linspace(0, F_s, N + 1)[:N/2]
    spl = 20. * numpy.log10(numpy.abs(H[:N/2]))
    phase = 180. / numpy.pi * numpy.angle(H[:N/2])

    return (freq, spl, phase)

#   Get the measured response
impulse_response = ff._read_ir_file('measurements/tweeterleft2.txt', impulse_box[0], impulse_box[1])
(t_ir, h_ir) = dict_to_arrays(impulse_response)
#   Clip measured response to a reasonable window
inds = (t_ir > -1e-3) * (t_ir < 1e-2)
t_ir = t_ir[inds]
h_ir = h_ir[inds]

#   Run the function to construct EQ filter
filt = ff.invert_measurement('measurements/tweeterleft2.txt', impulse_box, freq_box)

#   Model the equalized response
h_comb = scipy.signal.convolve(h_ir, filt.coefficients)
N_comb = h_comb.shape[0]
t_comb = numpy.linspace(0, dt * (N_comb - 1), N_comb)

#   Plot 1: IR
t_filt = numpy.linspace(0, dt * (N_filt - 1), N_filt)
pyplot.figure(figsize=(18, 7))
pyplot.subplot(1, 2, 1)
pyplot.hold(True)
pyplot.plot(t_filt, normalize_ir(filt.coefficients), 'b-')
pyplot.plot(t_ir, normalize_ir(h_ir), 'r-')
pyplot.plot(t_comb, normalize_ir(h_comb), 'g-')
pyplot.xlim([-1e-3, 4e-3])
pyplot.legend(['Filt', 'Meas', 'Comb'], loc='upper right')
pyplot.grid(True)
pyplot.title('Impulse responses')
#pyplot.savefig('test_inv_ir.pdf')

#   Plot 2: FR
#pyplot.figure()
pyplot.subplot(1, 2, 2)
pyplot.hold(True)
(freq_filt, spl_filt, phase_filt) = ir_to_fr(filt.coefficients)
(freq_ir, spl_ir, phase_ir) = ir_to_fr(h_ir)
(freq_comb, spl_comb, phase_comb) = ir_to_fr(h_comb)
pyplot.semilogx(freq_filt, normalize_fr(freq_filt, spl_filt, freq_box), 'b-')
pyplot.semilogx(freq_ir, normalize_fr(freq_ir, spl_ir, freq_box), 'r-')
pyplot.semilogx(freq_comb, normalize_fr(freq_comb, spl_comb, freq_box), 'g-')
pyplot.legend(['Filt', 'Meas', 'Comb'], loc='upper left')
pyplot.xlabel('Freq (Hz)')
pyplot.ylabel('Mag (dB)')
#pyplot.xlim([20, 20000])
pyplot.xlim([5e2, 2e4])
pyplot.ylim([-10, 5])
pyplot.grid(True)
pyplot.title('Frequency responses')
#pyplot.savefig('test_inv_fr.pdf')
pyplot.savefig('test_inv_multiplot.pdf')

pyplot.show()
