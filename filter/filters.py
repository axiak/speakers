import os
import gzip
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
        pyplot.xlim(0, 0.003)
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

    def _read_ir_file(self, filename, start_window=None, stop_window=None):
        impulse_response = {}
        if filename.endswith('.gz'):
            f = gzip.GzipFile(filename)
        else:
            f = open(filename)
        try:
            start_time = 0
            sample_interval = 0
            if start_window is None:
                start_window = start_time
            if stop_window is None:
                stop_window = 1e20

            for line in f:
                if 'Sample interval' in line:
                    sample_interval = float(line.split()[0])
                elif 'Start time' in line:
                    start_time = float(line.split()[0])
                if not line.strip():
                    break
            now = start_time
            for line in f:
                if not line.strip():
                    return impulse_response
                if start_window <= now <= stop_window:
                    impulse_response[now] = float(line.strip())
                else:
                    impulse_response[now] = 0
                now += sample_interval
            return impulse_response
        finally:
            f.close()

    def _mask_freq_response(self, sample_rate, fft_coefs, start_freq, stop_freq):
        #   When masking the frequency response, we have to keep in mind
        #   that the frequency scale wraps around.  It starts at 0, goes up to
        #   Fs / 2, then wraps around to -Fs / 2 and comes back up to -0.
        #   Both the positive and negative sides of the scale need to be masked.
        N = fft_coefs.shape[0]
        freq_scale = numpy.linspace(0, sample_rate, N + 1)[:-1]
        freq_scale[freq_scale > sample_rate / 2] -= sample_rate

        #   Snap start/end to freq scale so no interpolation is needed.
        #   Should be close enough if the FFT has a lot of points.
        coef_at_start = fft_coefs[int(start_freq * N / sample_rate)]
        coef_at_stop = fft_coefs[int(stop_freq * N / sample_rate)]

        #   Mask coefficients outside the desired band at both positive and negative frequency.
        fft_coefs[freq_scale < -stop_freq] = coef_at_stop
        fft_coefs[(freq_scale > -start_freq) * (freq_scale < start_freq)] = coef_at_start
        fft_coefs[freq_scale > stop_freq] = coef_at_stop

        return fft_coefs

    def _build_freq_response(self, impulse_response, start_freq, stop_freq):
        items = sorted(impulse_response.items())
        sample_rate = 1 / (items[1][0] - items[0][0])
        freq_scale = numpy.linspace(0, sample_rate, len(items) + 1)
        freq_scale[freq_scale > sample_rate / 2] -= sample_rate

        fft = scipy.fftpack.fft(numpy.array([item[1] for item in items]))
        return freq_scale[:-1], self._mask_freq_response(sample_rate, fft, start_freq, stop_freq)

    def _read_spl_file(self, filename):
        db_dict = {}
        with open(filename) as f:
            for line in f:
                if line.startswith('*') or not line.strip():
                    continue
                freq, db = [float(x) for x in line.split()[:2]]
                # Below 39hz the measurements are of the box and
                # cannot be corrected without overflow
                if freq >= 60:
                    db_dict[freq] = db
        return {freq: 10**(db / 20.0) for freq, db in db_dict.items()}

    @filter_cache
    def invert_measurement(self, filename, impulse_box, freq_box, name=None):
        impulse_response = self._read_ir_file(filename, impulse_box[0], impulse_box[1])
        sample_freq, new_fft = self._build_freq_response(impulse_response, freq_box[0], freq_box[1])

        #   Invert the measured FFT magnitude to get the filter FFT magnitude
        filt_fft_mag = numpy.abs(new_fft) ** -1
        #   Make the filter minimum phase using the Hilbert transform
        filt_fft_phase = -numpy.imag(scipy.signal.hilbert(numpy.log(filt_fft_mag)))
        #   Construct the filter impulse response from the FFT magnitude and phase
        min_phase_fft = numpy.exp(filt_fft_phase * 1j) * filt_fft_mag
        min_phase_ifft = scipy.fftpack.ifft(min_phase_fft)
        
        coefs = numpy.real(min_phase_ifft[:self.filter_size])
        coefs /= numpy.sqrt(numpy.sum(coefs ** 2))
        
        f = Filter(
            'invert_measurement',
            name,
            self.sample_freq,
            coefs
        )
        f.phase = filt_fft_phase
        f.min_phase_fft = min_phase_fft
        f.min_phase_ifft = min_phase_ifft
        return f

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
        if 0 not in signal:
            signal[0] = min(signal.items())[1]
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
