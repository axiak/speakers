import os
import numpy
import shelve
import pickle
import scipy.fftpack
import scipy.signal
from functools import wraps
from matplotlib import pyplot


class Filter(object):
    sample_freq = None
    coefficients = None
    name = None
    filter_type = None

    def __init__(self, filter_type, name, sample_freq, coefficients):
        self.filter_type = filter_type
        self.name = name
        self.sample_freq = sample_freq
        self.coefficients = coefficients

    def __add__(self, other):
        self.__assert_compatible(other)
        return Filter(
            "{0}_add_{1}".format(*tuple(sorted([self.filter_type, other.filter_type]))),
            "{0} + {1}".format(self.name, other.name),
            self.sample_freq,
            self.coefficients + other.coefficients
        )

    def __mul__(self, other):
        self.__assert_compatible(other)
        coefs = scipy.signal.convolve(self.coefficients, other.coefficients, mode='same')

        return Filter(
            "{0}_conv_{1}".format(*tuple(sorted([self.filter_type, other.filter_type]))),
            "{0} * {1}".format(self.name, other.name),
            self.sample_freq,
            coefs
        )

    def __assert_compatible(self, other):
        if self.sample_freq != other.sample_freq:
            raise Exception("Cannot add two filters of different sample rate: {0} =/= {1}".format(
                self, other
            ))
        if len(self.coefficients) != len(self.coefficients):
            raise Exception("Cannot add two filters of different size: {0} =/= {1}".format(
                self, other
            ))

    def plot(self, output_file=None, prompt=True):
        filter_coefs = self.coefficients
        t = numpy.linspace(0, float(len(filter_coefs - 1)) / self.sample_freq, len(filter_coefs))
        fig = pyplot.figure(figsize=(10, 10))

        pyplot.subplot(211)

        pyplot.plot(t, numpy.abs(filter_coefs))
        pyplot.plot(t, numpy.real(filter_coefs))
        pyplot.legend(['abs', 'real'], loc='upper right')

        pyplot.grid(True)
        pyplot.xlabel('Time (s)')
        pyplot.ylabel('Filter coefficient')
        pyplot.title('Filter impulse response')

        ax1 = fig.add_subplot(212)
        (freq, ampl) = scipy.signal.freqz(self.coefficients)
        freq_hz = (freq * self.sample_freq / (2 * numpy.pi))[1:]

        pyplot.semilogx(freq_hz, 20 * numpy.log10(numpy.abs(ampl[1:])), 'b-')
        pyplot.grid(True)
        pyplot.ylabel('Amplitude (dB)')
        pyplot.legend(['Amplitude', 'Phase'], loc='upper right')

        ax1.twinx()
        pyplot.semilogx(freq_hz, 180 / numpy.pi * numpy.unwrap(numpy.angle(ampl[1:])), 'r-')
        pyplot.ylabel('Phase angle (deg)')
        pyplot.xlabel('Frequency (Hz)')
        pyplot.legend(['Phase'], loc='lower right')
        pyplot.title('Filter frequency response')

        if output_file:
            pyplot.savefig(output_file)
        else:
            if prompt:
                print 'Please close figure window to continue'
            pyplot.show()

    def __repr__(self):
        return '<{0} filter for {1}hz sample freq with {2} taps{3}>'.format(
            self.filter_type,
            self.sample_freq,
            len(self.coefficients),
            self.name and ': ' + self.name or ''
        )

    def _repr_html_(self):
        self.plot(prompt=False)


__filter_db = shelve.open(os.path.join(os.path.dirname(__file__), 'filter_cache'))


def filter_cache(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        method_name = method.__name__
        if kwargs.pop('nocache', False):
            return method(*args, **kwargs)
        self, args = args[0], args[1:]
        factory_args = sorted(self.__dict__.items())
        key = pickle.dumps((method_name, factory_args, args, sorted(kwargs.items())))
        cached_value = __filter_db.get(key)
        if cached_value is None:
            cached_value = method(self, *args, **kwargs)
            __filter_db[key] = cached_value
            __filter_db.sync()
        return cached_value
    return wrapper


class FilterFactory(object):
    sample_freq = None
    filter_size = None

    def __init__(self, sample_freq, filter_size):
        self.sample_freq = sample_freq
        self.filter_size = filter_size

    @filter_cache
    def allpass(self):
        return Filter(
            'allpass',
            None,
            self.sample_freq,
            scipy.fftpack.ifft(numpy.repeat(1, self.filter_size))
        )

    @filter_cache
    def nopass(self):
        return Filter(
            'nopass',
            None,
            self.sample_freq,
            numpy.repeat(0, self.filter_size)
        )

    @filter_cache
    def remez_filter(self, inner_bands, band_coefficients, grid_density=256, name=None):
        return Filter(
            'remez',
            name,
            self.sample_freq,
            scipy.signal.remez(
                self.filter_size,
                [0] + inner_bands + [self.sample_freq / 2],
                band_coefficients,
                Hz=self.sample_freq,
                maxiter=200,
                grid_density=256
            ))

    @filter_cache
    def butter_filter(self, order, cutoffs, btype, name=None):
        ba = scipy.signal.butter(order,
                                 2 * numpy.pi * numpy.array(cutoffs),
                                 btype=btype,
                                 analog=True,
                                 output='ba'
                                 )

        t = numpy.linspace(0, float(self.filter_size - 1) / self.sample_freq, self.filter_size)

        coefs = scipy.signal.impulse(ba, T=t)[1]

        coefs /= float(self.sample_freq)  # TODO - What sort of normalization do we need here?
        return Filter(
            'butterworth',
            name,
            self.sample_freq,
            coefs
        )

    @filter_cache
    def hilbert_filter(self, signal, name=None):
        coefs = scipy.signal.hilbert(signal)
        #coefs /= coefs.sum()
        coefs = scipy.fftpack.ifft(numpy.imag(coefs))
        return Filter(
            'hilbert',
            name,
            self.sample_freq,
            coefs
        )

    def optimization_filter(self, signal, optimization_order=None, name=None, disp=False, maxIter=100):
        if optimization_order is None:
            optimization_order = self.filter_size * 4

        if isinstance(signal, dict):
            signal_func = self.__signal_from_dict(signal)
        else:
            signal_func = self.__signal_from_center(signal)

        target_response = signal_func(numpy.linspace(0, self.sample_freq, optimization_order))
        target_no_dc = target_response[1:]

        def cost_function(vector):
            freq, ampl = scipy.signal.freqz(numpy.real(vector), worN=len(target_response))
            cost = max(numpy.abs(target_no_dc - numpy.abs(freq[1:])))
            # Todo - minimize phase changes
            return cost

        result = scipy.optimize.minimize(cost_function,
                                         scipy.fftpack.ifft(target_response),
                                         options=dict(
                                             disp=disp,
                                             maxiter=maxIter
                                         ))

        f = Filter(
            'testing',
            name,
            self.sample_freq,
            result.x
        )
        f.result = result
        return f

    def spectral_factorization(self, coefficients):
        """
        See http://cvxr.com/cvx/examples/filter_design/html/spectral_fact.html
        """
        n = len(coefficients)
        multiple = 100
        m = self.filter_size * multiple

        w = (self.sample_freq * numpy.arange(0, m) / m).reshape(-1, 1)

        R = numpy.hstack([
            numpy.ones((m, 1)),
            2 * numpy.cos(numpy.kron(w, numpy.arange(1, n).reshape(-1, 1).transpose()))
        ]) * numpy.array(coefficients)

        alpha = 1 / 2.0 * numpy.log(R)

        # hibert transform
        alpha_tmp = scipy.fftpack.fft(alpha)
        alpha_tmp[int(m / 2.0) + 1:m] = -alpha_tmp[int(m / 2.0) + 1:m]
        alpha_tmp[1] = 0
        alpha_tmp[int(m / 2.0) + 1] = 0
        phi = numpy.real(scipy.fftpack.ifft(1j * alpha_tmp))

        indexes = numpy.array([i for i in range(m)
                               if i % multiple == 0])
        alpha1 = alpha[indexes]
        phi1 = phi[indexes]

        return numpy.real(scipy.fftpack.ifft(numpy.exp(alpha1 + 1j * phi1), n))[0]

    def __signal_from_dict(self, signal):
        if self.sample_freq / 2.0 not in signal:
            signal[self.sample_freq / 2.0] = max(signal.items())[1]
        items = signal.items()

        return self.__symmetric_signal_function(scipy.interpolate.interp1d(
            [item[0] for item in items],
            [item[1] for item in items]
        ))

    def __signal_from_center(self, signal):
        return self.__symmetric_signal_function(scipy.interpolate.interp1d(
            numpy.linspace(0, self.sample_freq / 2., len(signal)),
            signal
        ))

    def __symmetric_signal_function(self, half_function):
        half_point = self.sample_freq / 2.

        def result(x):
            if x <= half_point:
                return half_function(x)
            else:
                return half_function(self.sample_freq - x)
        return numpy.vectorize(result)

    def close(self):
        __filter_db.sync()
