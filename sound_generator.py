import numpy as np
from numpy.lib.twodim_base import triu
import scipy.integrate
from scipy.io import wavfile, loadmat
import math
from scipy.io.wavfile import write


class SoundGenerator:

    def __init__(self, mat_file_path: str, gen_mat_file_path: str):
        self.mat_file_path     = mat_file_path
        self.gen_mat_file_path = gen_mat_file_path

        gen_mat_dic = loadmat(self.gen_mat_file_path)

        self.a = gen_mat_dic['a']
        self.b = gen_mat_dic['b']
        self.phi = gen_mat_dic['phi']
        self.omega = gen_mat_dic['omega']

        # Read sound file, know the length
        sound_file_path = self.mat_file_path.replace('mat', 'wav')
        Fs, sound_data = wavfile.read(sound_file_path)
        y_data = sound_data[:, 0]
        t_data = np.linspace(0 + 1 / Fs, 1 / Fs * y_data.size, y_data.size)

        self.y = y_data
        self.t = t_data

        y_new = np.zeros(t_data.size)

        for i in range(0, 7):
            y_new = y_new + self.a[0, i] * np.exp(self.b[0, i] * self.t) * np.sin(2 * np.pi * self.omega[0, i] * self.t + self.phi[0, i] * np.pi)

        scaled = np.int16(y_new / np.max(np.abs(y_new)) * 32767)

        write(self.gen_mat_file_path.replace('mat', 'wav'), Fs, scaled)
