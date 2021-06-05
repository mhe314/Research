import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim
from scipy.io import wavfile, savemat
from scipy.fftpack import fft
from scipy.signal import find_peaks, stft
from scipy.optimize import curve_fit


class FeatureExtractor:

    def __init__(self, sound_file_path: str, plot: bool = False):

        self.sound_file_path = sound_file_path  # this should be a .wav file

        # Read sound file
        Fs, sound_data = wavfile.read(self.sound_file_path)
        y_data = sound_data[:, 0]
        t_data = np.linspace(0 + 1 / Fs, 1 / Fs * y_data.size, y_data.size)

        # FFT
        N = y_data.size
        dt = 1 / Fs
        t = dt * np.linspace(0, N - 1, num=N)
        dF = Fs / N
        f = dF * np.linspace(0, N / 2 - 1, num=round(N / 2))
        X = fft(y_data) / N
        X = X[0:round(N / 2)]
        X[1:] = 2 * X[1:]
        X = abs(X)

        if plot:
            # Plot FFT
            plt.plot(f, X)
            xlim(0, 3000)  # Define x axis limitation in the figure
            plt.grid()
            plt.show()

        # Find fundamental frequencies
        Index = np.argmax(X)
        TF = find_peaks(X, height=None, threshold=None, distance=round(Index))
        TF = TF[0]
        temp1 = f[TF]
        self.omega = temp1[0:8]

        # short-time fourier transform
        f, t, s = stft(y_data, Fs, window='boxcar', nperseg=2048 * 2, noverlap=None, nfft=None, detrend=False, return_onesided=True, )

        if plot:
            f_plot = f[0:400]
            s_plot = np.log(np.abs(s[1:400, ]))
            plt.pcolormesh(t, f_plot, s_plot)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

        # Find initial guess of a and b
        self.a = np.zeros(8)
        self.b = np.zeros(8)
        for i in range(0, 7):
            Index = np.argmin(np.abs(f - self.omega[i]))
            amp = np.abs(s[Index,])
            popt, pcov = curve_fit(self.func1, t, amp)
            self.a[i] = popt[0]
            self.b[i] = popt[1]
        c_ini = np.random.random(8)
        # p=np.concatenate((a, b,c_ini)) uncomment if you want to optimize all the features
        p = c_ini

        popt, pcov = curve_fit(self.func3, t_data, y_data, p)

        self.phi = popt
        # for i in range(0, 7): #uncommend if you want to optimize all the features
        # a_out[i]=popt[i]
        # b_out[i]=popt[i+8]
        # c_out[i]=popt[i+16]

        self.save_features()

    # Define fit function: optimize phase angles
    def func1(self, x):

        return self.a * np.exp(self.b * x)

    def func2(self, t, c0, c1, c2, c3, c4, c5, c6, c7):

        f = self.a[0] * np.exp(self.b[0] * t) * np.sin(self.omega[0] * 2 * np.pi * t + c0) + self.a[1] * np.exp(self.b[1] * t) * np.sin(
            self.omega[1] * 2 * np.pi * t + c1) + self.a[
                2] * np.exp(self.b[2] * t) * np.sin(self.omega[2] * 2 * np.pi * t + c2) + self.a[3] * np.exp(self.b[3] * t) * np.sin(
            self.omega[3] * 2 * np.pi * t + c3) + self.a[
                4] * np.exp(self.b[4] * t) * np.sin(self.omega[4] * 2 * np.pi * t + c4) + self.a[5] * np.exp(self.b[5] * t) * np.sin(
            self.omega[5] * 2 * np.pi * t + c5) + self.a[
                6] * np.exp(self.b[6] * t) * np.sin(self.omega[6] * 2 * np.pi * t + c6) + self.a[7] * np.exp(self.b[7] * t) * np.sin(
            self.omega[7] * 2 * np.pi * t + c7)
        return f

    def func3(self, t, c0, c1, c2, c3, c4, c5, c6, c7):

        return self.func2(t, c0, c1, c2, c3, c4, c5, c6, c7)

    def save_features(self):

        mat_dic = {"a": self.a, "b": self.b, "phi": self.phi, "omega": self.omega}
        save_feature_path = self.sound_file_path.replace('wav', 'mat')
        savemat(save_feature_path, mat_dic)
