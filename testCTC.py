## Import package
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import namedtuple
import unittest
import torch
# from ctcdecode import ctcdecode
import ctcdecode
from ctcdecode import CTCBeamDecoder

import Levenshtein

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class WSJ():
    """ Load the WSJ speech dataset

        Ensure WSJ_PATH is path to directory containing
        all data files (.npy) provided on Kaggle.

        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)

    """

    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'wsj0_dev')
        return self.dev_set

    os.environ["WSJ_PATH"] = '/home/ubuntu/CTCspeech/hw3p2-data-V2'

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'wsj0_train')
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (
                np.load(os.path.join(os.environ['WSJ_PATH'], 'transformed_test_data.npy'), encoding='bytes'), None)
        return self.test_set


def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'),
        np.load(os.path.join(path, '{}_merged_labels.npy'.format(name)), encoding='bytes')
    )


PHONEME_MAP = [
    ' ',  # "Blank"
    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
]


class SpeechDatasets(Dataset):
    ''' the customized datasets class '''

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x, y = torch.tensor(self.data[index]), torch.tensor((self.label[index]))
        return x, y

    def __len__(self):
        return len(self.data)


def padding(batch):
    ''' the padding function for the usage of collate function'''

    # 1. Sort
    sorted_pairs = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sorted_sequences = [x[0] for x in sorted_pairs]

    # 2. Pad sequence
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sorted_sequences, batch_first=True)
    length = torch.LongTensor([len(x) for x in sorted_sequences])

    #     labels = [x[1] + 1 for x in sorted_pairs]
    #     labels_length = torch.LongTensor([len(x) for x in labels])

    return sequences_padded, length, np.array([1, 1]), np.array([1, 1])


class CTCSpeech(nn.Module):
    '''Build the model with LSTM with 40 dimension input, 256 dimension hidden
        ,and 3 stacks. Additionally, add an MLP to generate outputs of size 47 '''

    def __init__(self):
        super(CTCSpeech, self).__init__()

        self.hidden_dim = 256
        self.embedding_dim = 40
        self.feature_size = 47

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=3,
                            bidirectional=True,
                            batch_first=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.feature_size)

    def forward(self, data, data_lengths, labels, labels_length):
        #         data, data_lengths, labels, labels_length = batch
        x_pack = pack_padded_sequence(data, data_lengths, batch_first=True)
        output, self.hidden = self.lstm(x_pack)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        output = self.hidden2tag(output[0])

        return output, self.hidden


def convert_to_string(tokens, vocab, seq_len):
    '''the function that convert vocabulary to sequence'''
    return ''.join([vocab[x] for x in tokens[0:seq_len]])


def testModel(model, test_loader, device):
    model.to(device)
    model.eval()

    with open('submission_1.txt', 'w') as file:
        with torch.no_grad():
            i = 1
            for batch_idx, (data, data_lengths, label, label_length) in enumerate(test_loader):
                label = torch.tensor(label)
                label_length = torch.tensor(label_length)

                data, data_lengths, label, label_length = \
                    data.to(device), data_lengths.to(device), label.to(device), label_length.to(device)

                outputs, hidden = model(data, data_lengths, label, label_length)

                # decode
                outputs_soft = outputs.permute(1, 0, 2)

                m = nn.Softmax(dim=2)
                outputs_soft = m(outputs_soft)

                probs_seq = outputs_soft
                decoder = ctcdecode.CTCBeamDecoder(PHONEME_MAP, beam_width=100,
                                                   blank_id=PHONEME_MAP.index(' '))
                beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)

                output_str = convert_to_string(beam_result[0][0], PHONEME_MAP, out_seq_len[0][0])

                file.write('\n' + '{}'.format(output_str))
                print(i)
                print('{}'.format(output_str))
                i += 1


def main():
    loader = WSJ()
    trainX, trainY = loader.train
    devX, devY = loader.dev
    testX = loader.test
    testX = np.array([i.astype(np.float32) for i in testX[0]])
    print('the length of training data is', testX[0].shape[0])
    testY = trainY[:(len(testX[0]))]

    #     trainDatasets = SpeechDatasets(trainX, trainY)
    #     valDatasets = SpeechDatasets(devX, devY)
    testDatasets = SpeechDatasets(testX, testY)
    # test_size = test_data.test_data.shape[0]

    #     train_loader = torch.utils.data.DataLoader(trainDatasets,
    #                                                shuffle=True,
    #                                                batch_size=batch_size,
    #                                                collate_fn=padding)

    #     val_loader = torch.utils.data.DataLoader(valDatasets,
    #                                              shuffle=True,
    #                                              batch_size=1,
    #                                              collate_fn=padding)

    test_loader = torch.utils.data.DataLoader(testDatasets,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=padding)

    speechmodel = CTCSpeech()

    criterion = nn.CTCLoss()

    optimizer = optim.Adam(speechmodel.parameters(), lr=0.001)
    # optimizer = optim.SGD(speechmodel.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_ft = torch.load('tf3.pt')

    testModel(model_ft, test_loader, device)


if __name__ == '__main__':
    main()
