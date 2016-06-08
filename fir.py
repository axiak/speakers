#!/usr/bin/env python

"""
FIR filtering demonstration.
Intended to prototype an embedded system that performs FFT based audio filtering.

Current capabilities:
- Play a WAV file through a lowpass filter.

To do:
- Generate more interesting filters (i.e. EQ rather than brickwall lowpass).
- Switch to a more causal (minimum phase?) filter design methodology.
- Implement and experiment with partitioned convolution to reduce latency.
"""

import sys
import numpy
import scipy.fftpack

import pyaudio
import scipy.io.wavfile
import scipy.signal
import time
from datetime import datetime

from matplotlib import pyplot

import pdb


def time_since(time_start):
    time_diff = datetime.now() - time_start
    time_float = time_diff.seconds + 1e-6 * time_diff.microseconds
    return time_float


def read_wav(filename):
    """
    Utility function for loading a WAV file.
    Wraps the scipy function, and normalizes format to float with range [-1, 1].
    """
    (F_s, x) = scipy.io.wavfile.read(filename)
    if x.dtype == numpy.int16:
        x = x.astype(float) / 32768.0
    else:
        raise Exception('Unsupported format %s.  Please implement.' % x.dtype)
    return (F_s, x)


class FilterPrep(object):
    """
    Base class for generating different types of filters.
    """

    def __init__(self, F_s, N_taps, *args, **kwargs):
        self.F_s = float(F_s)
        self.N = int(N_taps)

    def gen_lp(self, F_c1, F_c2, debug=False):
        #   Note: 'remez' is the scipy implementation of the Parks-McClellan algorithm.
        if F_c1 > 0:
            self.coefs = scipy.signal.remez(
                self.N,
                [0, F_c1 * 0.9, F_c1 * 1.1, F_c2 * 0.9, F_c2 * 1.1, self.F_s / 2],
                [0.0, 1.0, 0.0],
                Hz=self.F_s,
                maxiter=50
            )
        else:
            self.coefs = scipy.signal.remez(
                self.N,
                [0, F_c2 * 0.9, F_c2 * 1.1, self.F_s / 2],
                [1.0, 0.0],
                Hz=self.F_s,
                maxiter=50
            )

        self.coefs[:] = .03

        if debug:
            t = numpy.linspace(0, (self.N - 1) / self.F_s, self.N)
            fig = pyplot.figure(figsize=(10, 10))

            pyplot.subplot(211)
            pyplot.plot(t, self.coefs)
            pyplot.grid(True)
            pyplot.xlabel('Time (s)')
            pyplot.ylabel('Filter coefficient')
            pyplot.title('Filter impulse response')

            fig.add_subplot(212)
            (freq, ampl) = scipy.signal.freqz(self.coefs)
            freq_hz = freq * self.F_s / (2 * numpy.pi)
            pyplot.semilogx(freq_hz, 20 * numpy.log10(numpy.abs(ampl)), 'b-')
            pyplot.grid(True)
            pyplot.ylabel('Amplitude (dB)')
            #ax2 = ax1.twinx()
            pyplot.semilogx(freq_hz, 180 / numpy.pi * numpy.unwrap(numpy.angle(ampl)), 'r-')

            #   pyplot.legend(['Amplitude', 'Phase'], loc='upper right')

            pyplot.xlabel('Frequency (Hz)')
            pyplot.ylabel('Phase angle (deg)')
            pyplot.title('Filter frequency response')

            print 'Please close figure window to continue'
            pyplot.show()

        return self.coefs


class OverlapSave(object):
    """
    "Streaming" convolution based on overlap/save algorithm.
    """

    def __init__(self, filt, N, N_ch=1):
        """
        - filt: Filter coefficients (length M)
        - N: size of FFT (must be larger than M, preferably at least 4x larger)
        """
        assert isinstance(filt, numpy.ndarray)

        #   Compute FFT of the filter with the supplied number of points
        self.N_ch = int(N_ch)
        self.N = int(N)
        self.M = filt.shape[0]
        self.step = self.N - self.M + 1

        self.filt_padded = numpy.zeros((N,), dtype=complex)
        self.filt_padded[:self.M] = filt
        self.F = numpy.repeat(numpy.atleast_2d(scipy.fftpack.fft(self.filt_padded)).T, N_ch, 1)

    def clear(self):
        self.counter = 0
        self.pending_samples = numpy.zeros((self.N, self.N_ch))
        self.input_record = []
        self.output_record = []
        self.output_buffer = numpy.zeros((0, 2))
        self.output_buffer_idx = 0

    def process(self, samples):
        """
        Returns the new output samples that are generated, based on the supplied input.
        May run multiple FFT chunks if enough input is supplied.
        """
        assert samples.shape[1] == self.N_ch
        N_proc = 0
        N_in = samples.shape[0]
        x = self.pending_samples
        output_chunks = []
        while N_proc < N_in:
            #   print 'Counter = %d N_in = %d N_proc = %d N = %d' % (self.counter, N_in, N_proc, self.N)
            if self.counter + (N_in - N_proc) >= self.N:
                #   See if we can fill the next chunk.  If so, process that chunk.
                x = self.pending_samples
                x[self.counter:] = samples[N_proc:N_proc + self.N - self.counter]
                X = scipy.fftpack.fft(x, axis=0)
                Y = X * self.F
                y = numpy.real(scipy.fftpack.ifft(Y, axis=0))
                output_chunks.append(y[self.M - 1:])
                N_proc += self.step
                self.pending_samples[:self.M - 1] = x[self.step:]
                self.counter = self.M - 1
                #   print '-- Processed next chunk, N_proc = %d, reset counter to %d' % (N_proc, self.counter)
            else:
                #   Otherwise, save pending samples.
                self.pending_samples[self.counter:self.counter + N_in - N_proc] = samples[N_proc:]
                self.counter += N_in - N_proc
                N_proc = N_in
                #   print '-- Saved leftover samples: counter = %d' % self.counter

        #   Save for later debugging
        self.input_record.append(samples)
        self.output_record += output_chunks

        #   Return result
        if len(output_chunks) > 0:
            return numpy.concatenate(output_chunks, axis=0)
        else:
            return numpy.zeros((0, self.N_ch))

    def debug_display(self, F_s):
        x_in = numpy.concatenate(self.input_record, axis=0)
        x_out = numpy.concatenate(self.output_record, axis=0)

        N_in = x_in.shape[0]
        N_out = x_out.shape[0]

        t_in = numpy.linspace(0, (N_in - 1) / F_s, N_in)
        t_out = numpy.linspace(0, (N_out - 1) / F_s, N_out)

        pyplot.figure(figsize=(15, 8))
        pyplot.subplot(211)
        pyplot.hold(True)
        pyplot.grid(True)
        pyplot.plot(t_in, x_in[:, 0], 'b-')
        pyplot.plot(t_out, x_out[:, 0], 'r--')
        pyplot.xlabel('Time (s)')
        pyplot.ylabel('Signal')
        pyplot.title('Input and output of overlap/save')
        print 'Please close figure window to continue'
        pyplot.show()

    def write_wav(self, F_s, wav_filename):
        x_out = numpy.concatenate(self.output_record, axis=0)
        scipy.io.wavfile.write(wav_filename, F_s, x_out)


class StreamingPlayer(object):
    """
        Nonblocking audio playback via PyAudio.
    """

    def __init__(self,
                 F_s,
                 N_ch,
                 device_index,
                 frames_per_buffer=1024,
                 stream_function=None,
                 chunk=16384):
        self.F_s = F_s
        self.N_ch = N_ch
        self.frames_per_buffer = frames_per_buffer
        self.p = pyaudio.PyAudio()
        self.device_index = device_index
        self.samples = numpy.zeros((chunk, N_ch), dtype=numpy.int32)
        self.write_counter = 0
        self.read_counter = 0
        self.chunk = chunk
        self.play_state = False
        self.stream_function = stream_function
        self.input_record = []
        self.output_record = []

    def __enter__(self):
        self.stream = self.p.open(
            format=pyaudio.paInt32,
            channels=self.N_ch,
            rate=self.F_s,
            output=True,
            input=bool(self.stream_function),
            stream_callback=self.callback,
            output_device_index=self.device_index,
            input_device_index=None,  # self.device_index,
            frames_per_buffer=self.frames_per_buffer
        )
        return self

    def __exit__(self, *args, **kwargs):
        self.stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        if self.stream_function:
            return self.stream_callback(in_data, frame_count, time_info, status)
        else:
            return self.wav_callback(in_data, frame_count, time_info, status)

    def stream_callback(self, in_data, frame_count, time_info, status):
        in_data_np = numpy.fromstring(in_data, dtype=numpy.int32)
        in_data_np = in_data_np.reshape((frame_count, 2), order='F')

        in_data_np = in_data_np.astype(numpy.float32) / float(1 << 31) * .7071

        out_data = self.stream_function(in_data_np)

        if out_data.shape[0] == 0:
            print "underflow"
            out_data = numpy.zeros((frame_count, 2))

        self.input_record.append(in_data_np)
        self.output_record.append(out_data)

        out_data = (out_data * (1 << 31)).astype(numpy.int32)

        out_data = out_data.reshape((-1,), order='F')
        return out_data.tostring(), pyaudio.paContinue

    def debug_display(self, F_s):
        x_in = numpy.concatenate(self.input_record, axis=0)
        x_out = numpy.concatenate(self.output_record, axis=0)

        N_in = x_in.shape[0]
        N_out = x_out.shape[0]

        t_in = numpy.linspace(0, (N_in - 1) / F_s, N_in)
        t_out = numpy.linspace(0, (N_out - 1) / F_s, N_out)

        pyplot.figure(figsize=(15, 8))
        pyplot.subplot(211)
        pyplot.hold(True)
        pyplot.grid(True)
        pyplot.plot(t_in, x_in[:, 0], 'b-')
        pyplot.plot(t_out, x_out[:, 0], 'r--')
        pyplot.xlabel('Time (s)')
        pyplot.ylabel('Signal')
        pyplot.title('Input and output of overlap/save')
        print 'Please close figure window to continue'
        pyplot.show()

    def wav_callback(self, in_data, frame_count, time_info, status):
        #   If there is no data, tell pyaudio to stop.
        if self.write_counter < self.read_counter + frame_count:
            self.stream.stop_stream()
            return ('', pyaudio.paComplete)

        #   Returns all available data.
        ind_start = self.read_counter % self.chunk
        ind_end = (self.read_counter + frame_count) % self.chunk
        if ind_end <= ind_start:
            data = numpy.concatenate([self.samples[ind_start:], self.samples[:ind_end]], axis=0)
        else:
            data = self.samples[ind_start:ind_end]

        self.read_counter += frame_count
        return (data.tostring(), pyaudio.paContinue)

    def play_samples(self, samples):
        """
        Accepts samples in floating point format.
        Array should be 2-D with second dimension matching number of channels.
        Converts to fixed-point for playback.
        """

        #   Wait until we have space for these samples.
        while self.write_counter - self.read_counter > self.chunk:
            time.sleep(0.01)

        #   Convert samples to int32 format.
        #samples_int = samples[:]
        samples_int = (samples * (1 << 31)).astype(numpy.int32)

        #   Copy the samples to the circular buffer.
        ind_start = self.write_counter % self.chunk
        if ind_start + samples.shape[0] <= self.chunk:
            self.samples[ind_start:ind_start + samples.shape[0]] = samples_int
        else:
            #   Split around end of buffer
            split_count = self.chunk - ind_start
            self.samples[ind_start:] = samples_int[:split_count]
            self.samples[:samples.shape[0] - split_count] = samples_int[split_count:]
        self.write_counter += samples.shape[0]

        #   Tell pyaudio to start playing if this is the first chunk.
        if not self.play_state:
            self.time_start = datetime.now()
            self.stream.start_stream()
            self.play_state = True


def play_wav(wav_filename):
    M = 1025
    N = 4 * (M - 1)
    (F_s, x) = read_wav(wav_filename)
    N_ch = x.shape[1]
    N_samples = x.shape[0]
    print 'Applying -3 dB scaling to prevent clipping'
    x = x * 0.7071
    filt = FilterPrep(F_s, M)
    coefs = filt.gen_lp(0, 100., debug=False)
    print coefs
    print 'Computed FIR filter coefficients'

    o = OverlapSave(coefs, N, N_ch)
    print 'Initialized overlap/save algorithm'

    print F_s, N_ch

    with StreamingPlayer(F_s, N_ch, device_index=1) as p:
        print 'Playing source file, %d samples (%.3f sec)' % (N_samples, float(N_samples) / F_s)

        #   Send samples through the filter to the player.
        o.clear()
        chunk = 1024
        c = 0
        max_samples = min(N_samples, 1e8)
        while c < max_samples:
            samples_in = x[c:c+chunk]
            samples_out = o.process(samples_in)
            #print "in " + str(samples_in.shape)
            #print "out " + str(samples_out.shape)
            #   print 'After %d input samples: Got %s new output samples' % (c + chunk, samples_out.shape,)
            p.play_samples(samples_out)
            c += chunk

    o.debug_display(F_s)

    print 'Done'


def stream_audio():
    M = 1025
    N = 4 * (M - 1)
    N_ch = 2
    F_s = 44100
    print 'Applying -3 dB scaling to prevent clipping'
    filt = FilterPrep(F_s, M)
    coefs = filt.gen_lp(0, 1000.)

    o = OverlapSave(coefs, N, N_ch)
    print 'Initialized overlap/save algorithm'

    o.clear()

    with StreamingPlayer(
            F_s,
            N_ch,
            device_index=0,
            stream_function=o.process,
            frames_per_buffer=1024
    ) as p:
        try:
            while True:
                import time
                time.sleep(1)
        except:
            p.debug_display(F_s)

print 'Computed FIR filter coefficients'


if __name__ == '__main__':
    """ Default action:
        - Read the WAV file supplied as a command line argument.
          The sampling frequency is automatically set to match the file.
        - Prepare a 1 kHz lowpass filter using the Parks-McClellan algorithm.
          The filter has 1024 taps by default.
        - Play the WAV file through the FIR filter.
    """
    play_wav(sys.argv[1])
    #stream_audio()
