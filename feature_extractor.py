import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim
from scipy.io import wavfile, savemat
from scipy.fftpack import fft
from scipy.signal import find_peaks, stft
from scipy.optimize import curve_fit
import streamlit as st
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FeatureExtractor:
    def __init__(self, sound_file_path: str, check1, check2):
        

        self.sound_file_path = sound_file_path  # this should be a .wav file

        # Read sound file
        Fs, sound_data = wavfile.read(self.sound_file_path)
        y_data = sound_data[:, 0]
        t_data = np.linspace(0 + 1 / Fs, 1 / Fs * y_data.size, y_data.size)
        y = y_data
        t = t_data

        # FFT
        N = y.size
        dt = 1 / Fs
        t = dt * np.linspace(0, N - 1, num=N)
        dF = Fs / N
        f = dF * np.linspace(0, N / 2 - 1, num=round(N / 2))
        X = fft(y) / N
        X = X[0:round(N / 2)]
        X[1:] = 2 * X[1:]
        X = abs(X)

        if check1:
            # Plot FFT
            st.title('Fast Fourier Transform Plot')
            plt.title('Fast Fourier Transform')
            plt.ylabel('Magnitude')
            plt.xlabel('Frequency [Hz]')
            plt.plot(f, X)
            xlim(0, 3000)  # Define x axis limitation in the figure
            plt.grid()
            plt.annotate('Dominant Frequencies', xy=(600,180), xytext=(750, 220), arrowprops=dict(facecolor='black', shrink=10),
            horizontalalignment='left', verticalalignment='top')
            st.pyplot()

        # Find fundamental frequencies
        Index = np.argmax(X)
        basic_f = round(f[Index])
        TF = find_peaks(X, height=None, threshold=None, distance=round(Index))
        TF = TF[0]
        temp1 = f[TF]
        self.omega = temp1[0:8]

        # short-time fourier transform
        f, t, s = stft(y, Fs, window='boxcar', nperseg=2048 * 2, noverlap=None, nfft=None, detrend=False, return_onesided=True, );

        if check2:
            st.title('Short-time Fourier Transform Plot')
            f_plot = f[0:400]
            s_plot = np.log(np.abs(s[1:400, ]))
            plt.pcolormesh(t, f_plot, s_plot)
            plt.title('Short-time Fourier Transform Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            st.pyplot()

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
    @staticmethod
    def func1(x, a, b):

        return a * np.exp(b * x)

    @staticmethod
    def func2(t, c0, c1, c2, c3, c4, c5, c6, c7, a, b, omega):
        f = a[0] * np.exp(b[0] * t) * np.sin(omega[0] * 2 * np.pi * t + c0) + a[1] * np.exp(b[1] * t) * np.sin(omega[1] * 2 * np.pi * t + c1) + a[
            2] * np.exp(b[2] * t) * np.sin(omega[2] * 2 * np.pi * t + c2) + a[3] * np.exp(b[3] * t) * np.sin(omega[3] * 2 * np.pi * t + c3) + a[
                4] * np.exp(b[4] * t) * np.sin(omega[4] * 2 * np.pi * t + c4) + a[5] * np.exp(b[5] * t) * np.sin(omega[5] * 2 * np.pi * t + c5) + a[
                6] * np.exp(b[6] * t) * np.sin(omega[6] * 2 * np.pi * t + c6) + a[7] * np.exp(b[7] * t) * np.sin(omega[7] * 2 * np.pi * t + c7)
        return f

    def func3(self, t, c0, c1, c2, c3, c4, c5, c6, c7):

        return self.func2(t, c0, c1, c2, c3, c4, c5, c6, c7, self.a, self.b, self.omega)

    def save_features(self):

        mat_dic = {"a": self.a, "b": self.b, "phi": self.phi, "omega": self.omega}
        save_feature_path = self.sound_file_path.name.replace('wav', 'mat')
        savemat(save_feature_path, mat_dic)
