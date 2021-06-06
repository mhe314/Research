import glob
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import streamlit as st
import os
import io
import scipy.integrate
from scipy.io import wavfile, loadmat
import math
from scipy.io.wavfile import write

from config import path_dataset, path_default_sound_file, path_model
from feature_extractor import FeatureExtractor

# TODO: need melody_generator.py: uploaded to collab right now

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

st.title('Sound Modification App')  # Title for streamlit app


# Grabbing sound file data
def get_user_data():

    flag = ['New dataset', 'Default dataset']
    use_new_data = st.selectbox('Choose a new dataset or use default dataset', flag, 1)

    # load dataset
    if use_new_data == 'New dataset':
        uploaded_file = st.file_uploader('Choose a CSV file', accept_multiple_files=False)
        FeatureExtractor(uploaded_file)

    else:
        FeatureExtractor(path_default_sound_file)


class MyDataset(Dataset):
    """Dataset for MDS method"""

    def __init__(self, dataset_path, data_type):
        super(MyDataset, self).__init__()
        self.feat_list = ['freq_out', 'amp_out', 'a_out', 'b_out']
        if data_type == 'train':
            self.piano_list = glob.glob(f'{dataset_path}/piano/train/*.mat')
        else:
            self.piano_list = glob.glob(f'{dataset_path}/piano/test/*.mat')
        self.guitar_list = self.parse_guitar_list()

        # without normalization
        self.feats_all_p = self.wrap_data(self.piano_list)  # all piano features
        self.feats_all_g = self.wrap_data(self.guitar_list)
        print('no norm', self.feats_all_p[0][0, :])

        # normalization
        self.parse_minmax_p()
        self.parse_minmax_g()
        self.feats_all_p_norm = self.normalize_p()
        self.feats_all_g_norm = self.normalize_g()

    def parse_guitar_list(self):
        """Parse guitar_list"""
        guitar_list = []
        for piano_path in self.piano_list:
            guitar_path = piano_path.replace('piano', 'guitar')
            guitar_list.append(guitar_path)
        return guitar_list

    def merge_feats(self, mat_data):
        """merge 4 features into 1 feature"""
        freq = mat_data['omega'].reshape(1, -1)
        phi = mat_data['phi']
        a = mat_data['a']
        b = mat_data['b']
        feats_all = np.concatenate([freq, phi, a, b], axis=0)
        return feats_all

    def wrap_data(self, file_list):
        """add all features into an array"""
        feats_all = np.zeros((len(self.piano_list), 4, 8))  # for each feats: 4*8 dimension
        for idx, file_path in enumerate(file_list):
            feats = self.merge_feats(scipy.io.loadmat(file_path))
            feats_all[idx, :, :] = feats
        return feats_all

    def parse_minmax_p(self):
        """parse minmax for piano data"""
        self.freq_max_p, self.freq_min_p = 8424.0, 440.0
        self.phi_max_p, self.phi_min_p = 1.0, 0.0
        self.a_max_p, self.a_min_p = 0.34, 0
        self.b_max_p, self.b_min_p = -2, -17
        return None

    def parse_minmax_g(self):
        """parse minmax for guitar data"""
        self.freq_max_g, self.freq_min_g = 8424.0, 440.0
        self.phi_max_g, self.phi_min_g = 1.0, 0.0
        self.a_max_g, self.a_min_g = 0.23, 0
        self.b_max_g, self.b_min_g = -1.1, -12
        return None

    def normalize_p(self):
        """normalize piano data"""
        norm_all = np.zeros((len(self.piano_list), 4, 8))
        norm_all[:, 0, :] = (self.feats_all_p[:, 0, :] - self.freq_min_p) / (self.freq_max_p - self.freq_min_p)
        norm_all[:, 1, :] = (self.feats_all_p[:, 1, :] - self.phi_min_p) / (self.phi_max_p - self.phi_min_p)
        norm_all[:, 2, :] = (self.feats_all_p[:, 2, :] - self.a_min_p) / (self.a_max_p - self.a_min_p)
        norm_all[:, 3, :] = (self.feats_all_p[:, 3, :] - self.b_min_p) / (self.b_max_p - self.b_min_p)
        return norm_all

    def normalize_g(self):
        """normalize guitar data"""
        norm_all = np.zeros((len(self.piano_list), 4, 8))
        norm_all[:, 0, :] = (self.feats_all_g[:, 0, :] - self.freq_min_g) / (self.freq_max_g - self.freq_min_g)
        norm_all[:, 1, :] = (self.feats_all_g[:, 1, :] - self.phi_min_g) / (self.phi_max_g - self.phi_min_g)
        norm_all[:, 2, :] = (self.feats_all_g[:, 2, :] - self.a_min_g) / (self.a_max_g - self.a_min_g)
        norm_all[:, 3, :] = (self.feats_all_g[:, 3, :] - self.b_min_g) / (self.b_max_g - self.b_min_g)
        return norm_all

    def inverse_piano(self, feats_each):
        """inverse to original range for piano"""
        feats_orig = np.zeros((4, 8))
        feats_orig[0, :] = feats_each[0, :] * (self.freq_max_p - self.freq_min_p) + self.freq_min_p
        feats_orig[1, :] = feats_each[1, :] * (self.phi_max_p - self.phi_min_p) + self.phi_min_p
        feats_orig[2, :] = feats_each[2, :] * (self.a_max_p - self.a_min_p) + self.a_min_p
        feats_orig[3, :] = feats_each[3, :] * (self.b_max_p - self.b_min_p) + self.b_min_p
        return feats_orig

    def inverse_guitar(self, feats_each):
        """inverse to original range for guitar"""
        feats_orig = np.zeros((4, 8))
        feats_orig[0, :] = feats_each[0, :] * (self.freq_max_g - self.freq_min_g) + self.freq_min_g
        feats_orig[1, :] = feats_each[1, :] * (self.phi_max_g - self.phi_min_g) + self.phi_min_g
        feats_orig[2, :] = feats_each[2, :] * (self.a_max_g - self.a_min_g) + self.a_min_g
        feats_orig[3, :] = feats_each[3, :] * (self.b_max_g - self.b_min_g) + self.b_min_g
        return feats_orig

    def __len__(self):
        return len(self.piano_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        piano_path = self.piano_list[idx]       # piano sound path
        key_name = piano_path.split('/')[-1].split('.')[0]  # this get the key name from the file
        piano_feats = self.feats_all_p_norm[idx]
        guitar_feats = self.feats_all_g_norm[idx]

        return piano_feats.astype(np.float32), guitar_feats.astype(np.float32), key_name


class SimpleNet(nn.Module):
    """
    This is your FFNN model.
    """

    def __init__(self, input_dim, output_dim):
        """
        input_dim: input dimension
        output_dim: output dimension
        """
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, output_dim)

    def forward(self, inputs):
        """Net forward"""
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class Configer:
    """Parameters for training"""
    epoch = 5000
    batch_size = 4
    lr = 0.001
    p_length = 8
    g_length = 8

    def __init__(self):
        super(Configer, self).__init__()


def guitar_feature_generator(dataset_path, key_name, plot: bool = False):
    """Generate predicted guitar features from piano features

    Args:
        dataset_path: [String] folder to save dataset, please name it as "dataset";
        key_name: [String] key name that you want to generate. Example: "A4"

    Returns:
        gen_guitar_feats: [List] contains predicted guitar features in a dict,
                          note that this part can be used to generate many guitar features,
                          so we use a list to store the guitar features.
    """

    config = Configer()
    net = SimpleNet(config.p_length, config.g_length)
    net.to(device)
    net.load_state_dict(torch.load(path_model))
    net.eval()

    res, res_true = [], []
    dataset_train = MyDataset(dataset_path, 'train')
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

    for step, (inputs, targets, key_names) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.reshape(inputs.shape[0], 4, -1)
        targets = targets.reshape(inputs.shape[0], 4, -1)
        outputs = net(inputs)

        gen_feats_batch = outputs.detach().cpu().numpy()
        targets_batch = targets.detach().cpu().numpy()
        inputs_batch = inputs.detach().cpu().numpy()

        for i in range(gen_feats_batch.shape[0]):
            if key_names[i] != key_name:
                continue

            pred_feats_norm = gen_feats_batch[i].reshape(4, 8)

            # inverse data to original range
            pred_feats = dataset_train.inverse_guitar(pred_feats_norm)
            true_feats_norm = targets_batch[i].reshape(4, 8)
            true_feats = dataset_train.inverse_guitar(true_feats_norm)
            inputs_feats_norm = inputs_batch[i].reshape(4, 8)
            inputs_feats = dataset_train.inverse_piano(inputs_feats_norm)

            d = {
                'key': key_names[i],
                'freq': pred_feats[0, :],
                'phi': pred_feats[1, :],
                'a': pred_feats[2, :],
                'b': pred_feats[3, :],
            }
            d_true = {
                'key': key_names[i],
                'freq': true_feats[0, :],
                'phi': true_feats[1, :],
                'a': true_feats[2, :],
                'b': true_feats[3, :],
            }
            res_true.append(d_true)
            res.append(d)

            # # plot results
            if plot:
                fig = plt.figure(figsize=(12, 5))
                ax1 = fig.add_subplot(1, 2, 1)
                lns1 = plt.plot(pred_feats[0, :], pred_feats[2, :], '^', label='Prediction (G)')
                lns2 = plt.plot(true_feats[0, :], true_feats[2, :], 'v', label='Ground Truth (G)')
                plt.xlabel('Frequency', fontsize=16)
                plt.ylabel('Amplitude', fontsize=16)
                ax2 = ax1.twinx()
                lns3 = plt.plot(inputs_feats[0, :], inputs_feats[2, :], 'o', c='g', label='Ground Truth (P)')
                lns = lns1 + lns2 + lns3
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc=0, fontsize=14)
                plt.title('Key: ' + key_names[i], fontsize=18)

                ax3 = fig.add_subplot(1, 2, 2)
                lns1 = plt.plot(pred_feats[1, :], pred_feats[3, :], '^', label='Prediction (G)')
                lns2 = plt.plot(true_feats[1, :], true_feats[3, :], 'v', label='Ground Truth (G)')
                plt.xlabel('Phase angle', fontsize=16)
                plt.ylabel('Dampping coefficient $b_i$', fontsize=16)
                ax4 = ax3.twinx()
                lns3 = plt.plot(inputs_feats[1, :], inputs_feats[3, :], 'o', c='g', label='Ground Truth (P)')
                lns = lns1 + lns2 + lns3
                labs = [l.get_label() for l in lns]
                ax3.legend(lns, labs, loc=0, fontsize=14)
                plt.title('Key: ' + key_names[i], fontsize=18)

                plt.tight_layout()
                plt.savefig(f'results/MDS_pred_{key_names[i]}.jpg', doi=300)
                
                st.pyplot()
    return res


get_user_data()
gen_guitar_feats = pd.DataFrame(guitar_feature_generator(path_dataset, 'B4'))   # list of dictionaries: each with 4 dictionary keys
st.dataframe(gen_guitar_feats)
