import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import streamlit as st
import scipy.integrate
from scipy.io import wavfile, loadmat
import math
from scipy.io.wavfile import write

from config import path_dataset, path_model
from feature_extractor import FeatureExtractor
from sound_generator import SoundGenerator

# TODO: need melody_generator.py: uploaded to collab right now

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

st.title('Synthesizer')  # Title for streamlit app
st.text('Welcome to the Sound Modification App')
st.text('In this app, the goal is to take your piano audio file as an input and using feature \nengineering transform it to a guitar sound.')
st.text('Follow the steps below for how to use the app!')
st.text('')
st.header('Steps to use Synthesizer')
st.text('1. Upload your piano key audio file (ex: A4.wav)')
st.text('2. Click the play button on the audio bar to hear the audio file you selected')
st.text('3. Click on the checkbox below to display the plot you want to view. You can select   \n  more than one at a time')
st.text('4. Once you select a plot, make sure to give the app a few moments to load the plot')
st.text('5. Click the last checkbox to display the audio bar for the generated guitar key \n  sound. Press the play button on the audio bar to hear the generated guitar sound!')
st.text('')
st.text('')
st.set_option('deprecation.showPyplotGlobalUse', False)

check1 = st.checkbox('Display Fast Fourier Transform Plot')
check2 = st.checkbox('Display Short-time Fourier Transform Plot')
check3 = st.checkbox('Display Features')
check4 = st.checkbox('Display Plot of Features')
check5 = st.checkbox('Display Generated Audio')
global key
key = 'A4'


# Grabbing sound file data
def get_user_data(check1, check2, check3, check4) -> bool:

    uploaded_file = st.file_uploader('Choose a sound file', accept_multiple_files=False)

    if uploaded_file:
        global key
        st.audio(uploaded_file)
        file = str(uploaded_file)
        #st.title(file)
        file = file[25:27]
        key = file.replace('.wav', '')
        FeatureExtractor(uploaded_file, check1, check2)
        return True

    return False
#st.title(key)


class MyDataset(Dataset):
    """Dataset for MDS method"""

    def __init__(self, path_dataset, data_type):
        super(MyDataset, self).__init__()
        self.feat_list = ['freq_out', 'phi_out', 'a_out', 'b_out']

        # TODO: make the files paths universal to access github files
        if data_type == 'train':
            # self.piano_list = glob.glob(glob.escape(f'{dataset_path}/piano/train/**/*.mat'))
            # self.piano_list = glob.glob(r'https://github.com/mhe314/Research/tree/master/piano/train/*.mat')
            self.piano_list = [r'piano/train/{}.mat'.format(key)]

            # self.piano_list = glob.glob('/**/*.mat', recursive=True)
        else:
            # self.piano_list = glob.glob(f'**/*{dataset_path}/piano/test/**/*.mat')
            # self.piano_list = glob.glob(r'https://github.com/mhe314/Research/tree/master/piano/test/*.mat')

            # self.piano_list = glob.glob(glob.escape('/piano/test/*.mat'))
            self.piano_list = [r'piano/test/{}.mat'.format(key)]
            # self.piano_list = glob.glob('/**/*.mat', recursive=True)

        self.guitar_list = self.parse_guitar_list()

        print(self.piano_list)

        # without normalization
        self.feats_all_p = self.wrap_data(self.piano_list)  # all piano features
        self.feats_all_g = self.wrap_data(self.guitar_list)
        # print('no norm', self.feats_all_p[0][0, :])

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


# def model_trainer(path_dataset):
#     '''train model
    
#     Args:
#         dataset_path: [String] folder to save dataset, please name it as "dataset";

#     Returns:
#         None, but save model to current_folder + "results/mode.pkl"
#     '''
#     # configeration
#     config = Configer()

#     dataset_train = MyDataset(path_dataset, 'train')
#     dataset_test = MyDataset(path_dataset, 'test')
#     print(f'[DATASET] The number of paired data (train): {len(dataset_train)}')
#     print(f'[DATASET] The number of paired data (test): {len(dataset_test)}')
#     print(f'[DATASET] Piano_shape: {dataset_train[0][0].shape}, guitar_shape: {dataset_train[0][1].shape}')

#     # dataset
#     train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
#     test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True)

#     net = SimpleNet(config.p_length, config.g_length)
#     net.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(net.parameters(), lr=config.lr)
#     scheduler = StepLR(optimizer, step_size=int(config.epoch/4.), gamma=0.3)

#     # Note that this part is about model_trainer
#     loss_list = []
#     for epoch_idx in range(config.epoch):
#         # train
#         for step, (piano_sound, guitar_sound, _) in enumerate(train_loader):
#             inputs = piano_sound.to(device)
#             targets = guitar_sound.to(device)
#             inputs = inputs.reshape(inputs.shape[0], 4, -1)
#             targets = targets.reshape(inputs.shape[0], 4, -1)

#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             loss_list.append(loss.item())
#             loss.backward()
#             optimizer.step()

#         # eval
#         if epoch_idx % int(config.epoch/10.) == 0:
#             net.eval()
#             for step, (inputs, targets, _) in enumerate(train_loader):
#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#                 inputs = inputs.reshape(inputs.shape[0], 4, -1)
#                 targets = targets.reshape(inputs.shape[0], 4, -1)
#                 outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             print(f'epoch: {epoch_idx}/{config.epoch}, loss: {loss.item()}')

#     # save model
#     torch.save(net.state_dict(), path_dataset.replace('dataset', 'results')+'/model.pkl')

#     # plot loss history
#     fig = plt.figure()
#     plt.plot(loss_list, 'k')
#     plt.ylim([0, 0.02])
#     plt.xlabel('Iteration', fontsize=16)
#     plt.ylabel('Loss', fontsize=16)
#     plt.tight_layout()
#     st.pyplot()
#     #plt.savefig('results-MDS/MDS_loss.jpg', doi=300)

# model_trainer(path_dataset)



def guitar_feature_generator(path_dataset, key_name, plot: bool = True):
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
    net.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    net.eval()

    res, res_true = [], []
    dataset_train = MyDataset(path_dataset, 'train')
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
                st.title(('key name[i]:', key_names[i], 'key name:', key_name))
                continue

            pred_feats_norm = gen_feats_batch[i].reshape(4, 8)

            # inverse data to original range
            pred_feats = dataset_train.inverse_guitar(pred_feats_norm)
            #st.title(pred_feats)
            #scipy.io.savemat('A4_generated.mat', pred_feats)
            true_feats_norm = targets_batch[i].reshape(4, 8)
            true_feats = dataset_train.inverse_guitar(true_feats_norm)
            inputs_feats_norm = inputs_batch[i].reshape(4, 8)
            inputs_feats = dataset_train.inverse_piano(inputs_feats_norm)

            d = {
                'Key': key_names[i],
                'Frequency [Hz]': pred_feats[0, :],
                'Phi [radians]': pred_feats[1, :],
                'Amplitude (a)': pred_feats[2, :],
                'Damping Coefficient (b)': pred_feats[3, :],
            }
            d_true = {
                'Key': key_names[i],
                'Frequency [Hz]': true_feats[0, :],
                'Phi [radians]': true_feats[1, :],
                'Amplitude (a)': true_feats[2, :],
                'Damping Coefficient (b)': true_feats[3, :],
            }
            res_true.append(d_true)
            res.append(d)

            # plot results
    if plot:
        #st.title('About to plot')  # Title for streamlit app
        fig = plt.figure(figsize=(12, 5))
        # st.pyplot(fig)
        ax1 = fig.add_subplot(1, 2, 1)
        lns1 = plt.plot(pred_feats[0, :], pred_feats[2, :], '^', label='Prediction (G)')
        lns2 = plt.plot(true_feats[0, :], true_feats[2, :], 'v', label='Ground Truth (G)')
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude, a', fontsize=16)
        ax2 = ax1.twinx()
        lns3 = plt.plot(inputs_feats[0, :], inputs_feats[2, :], 'o', c='g', label='Ground Truth (P)')
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0, fontsize=14)
        plt.title('Key: ' + key_names[i], fontsize=18)

        ax3 = fig.add_subplot(1, 2, 2)
        lns1 = plt.plot(pred_feats[1, :], pred_feats[3, :], '^', label='Prediction (G)')
        lns2 = plt.plot(true_feats[1, :], true_feats[3, :], 'v', label='Ground Truth (G)')
        plt.xlabel('Phase Angle [radians]', fontsize=16)
        plt.ylabel('Damping Coefficient, $b_i$', fontsize=16)
        ax4 = ax3.twinx()
        lns3 = plt.plot(inputs_feats[1, :], inputs_feats[3, :], 'o', c='g', label='Ground Truth (P)')
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc=0, fontsize=14)
        plt.title('Key: ' + key_names[i], fontsize=18)

        plt.tight_layout()
        plt.show()
        #st.pyplot()
        #st.title('Plotted')  # Title for streamlit app

        # plt.savefig(f'results/MDS_pred_{key_names[i]}.jpg', doi=300)

    return res


if get_user_data(check1, check2, check3, check4):
    # TODO: change the key name (currently it is "A4")
    gen_guitar_feats = pd.DataFrame(guitar_feature_generator(path_dataset, key))   # list of dictionaries: each with 4 dictionary keys
    if check3: 
        st.title('Features')
        st.table(gen_guitar_feats)
    if check4: 
        st.title('Plot of Features')
        st.pyplot()
        
    if check5:
        st.title('Generated Audio')
        st.audio('guitar/train/{}.wav'.format(key))
        #generated = SoundGenerator(path_dataset)
        #st.audio(generated)
    
