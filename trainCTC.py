## Import package
import numpy as np
import pandas as pd
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils,datasets, models
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

import Levenshtein as Levenshtein

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

N_STATES = 138
N_PHONEMES = N_STATES // 3
PHONEME_LIST = [
    " ",
    "+BREATH+",
    "+COUGH+",
    "+NOISE+",
    "+SMACK+",
    "+UH+",
    "+UM+",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "SIL",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH"
]

PHONEME_MAP = [
    ' ',
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

assert len(PHONEME_LIST) == len(PHONEME_MAP)
assert len(set(PHONEME_MAP)) == len(PHONEME_MAP)

os.environ["WSJ_PATH"] = '/home/ubuntu/CTCspeech/hw3p2-data-V2'


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



class SpeechDatasets(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label= label

    def __getitem__(self, index):
        x, y = torch.tensor(self.data[index]),torch.tensor((self.label[index]))
        return x,y

    def __len__(self):
        return len(self.data)


def padding(batch):
    # 1. Sort
    sorted_pairs = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sorted_sequences = [x[0] for x in sorted_pairs]

    # 2. Pad sequence
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sorted_sequences, batch_first=True)
    length = torch.LongTensor([len(x) for x in sorted_sequences])

    labels = [x[1] + 1 for x in sorted_pairs]
    labels_length = torch.LongTensor([len(x) for x in labels])

    return sequences_padded, length, labels, labels_length


class CTCSpeech(nn.Module):
    def __init__(self):
        super(CTCSpeech, self).__init__()

        self.hidden_dim = 256
        self.embedding_dim = 40
        self.feature_size = 47

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=3,
                            bidirectional=True,
                            batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.feature_size)

    #         self.hidden = self.init_hidden()

    def forward(self, data, data_lengths, labels, labels_length):
        #         data, data_lengths, labels, labels_length = batch
        x_pack = pack_padded_sequence(data, data_lengths, batch_first=True)
        output, self.hidden = self.lstm(x_pack)
        output = nn.utils.rnn.pad_packed_sequence(output)
        output = self.hidden2tag(output[0])

        return output, self.hidden

def convert_to_string(tokens, vocab, seq_len):
    return ''.join([vocab[x] for x in tokens[0:seq_len]])


def train_epoch(model, train_loader, val_loader, criterion, optimizer, epochs, train_size, val_size):
    ini_time = time.time()

    # initial weight
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # metrics record statistics
    metrics = []

    # loop through each epoch
    for epoch in range(epochs):
        # set model to train model
        model.train()
        model.to(device)

        #         scheduler.step()

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # initialize the running loss to 0
        epoch_loss = 0.0
        correct = 0
        start_time = time.time()

        for batch_idx, (data, data_lengths, label, label_length) in enumerate(train_loader):
            print(batch_idx)

            #             data, data_lengths, label, label_length  = batch
            label_pack = torch.cat(label)

            data, data_lengths, label_pack, label_length = \
                data.to(device), data_lengths.to(device), label_pack.to(device), label_length.to(device)

            # refresh the parameter gradients
            optimizer.zero_grad()
            if batch_idx % 2 == 0:
                print(' Progress %s: %d/%d' % (epoch, batch_idx, len(train_loader)))

            # forward + backward + optimize
            outputs, hidden = model(data, data_lengths, label, label_length)
            outputs = outputs.log_softmax(2).detach().requires_grad_()

            loss = criterion(outputs, label_pack, data_lengths, label_length)

            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

            # end of an epoch
        end_time = time.time()
        print('Epoch %d Training Loss: ' % (epoch + 1), epoch_loss, 'Time: ', end_time - start_time, 's')

        #         # print statistics
        #         total_loss = epoch_loss/train_size
        #         train_error = 1.0 - correct/train_size
        #         train_acc = correct/train_size

        #         # validation process
        #         val_correct = 0
        val_distance = 0
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, data_lengths, label, label_length) in enumerate(val_loader):
                data, data_lengths, label_pack, label_length = \
                    data.to(device), data_lengths.to(device), label_pack.to(device), label_length.to(device)

                outputs, hidden = model(data, data_lengths, label, label_length)
                outputs = outputs

                m = nn.Softmax()
                outputs = m(outputs)

                probs_seq = outputs
                decoder = ctcdecode.CTCBeamDecoder(PHONEME_MAP, beam_width=100,
                                                   blank_id=PHONEME_MAP.index(' '))
                beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
                output_str = convert_to_string(beam_result[0][0], self.vocab_list, out_seq_len[0][0])

                true_str = ''.join([PHONEME_MAP[i] for i in label])
                ld = Levenshtein()
                distance = ld.distance(output_str, true_str)
                val_distance += distance

            val_avg_distance = val_distance / val_size

            # record best weights
            if val_avg_distance < best_acc:
                best_acc = val_avg_distance
                best_model_wts = copy.deepcopy(model.state_dict())

    # end of total training
    time_elapsed = time.time() - ini_time

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def main():
    loader = WSJ()
    trainX, trainY = loader.train
    devX, devY = loader.dev

    trainX, trainY = trainX[:7], trainY[:7]
    devX, devY = devX[:7], devY[:7]

    testX = loader.test
    print('the length of training data is',trainX.shape[0])
    print('the length of validation data is',devX.shape[0])
    testY = [np.array([0]) for i in range(len(testX[0]))]

    trainDatasets = SpeechDatasets(trainX, trainY)
    valDatasets = SpeechDatasets(devX, devY)
    testDatasets = SpeechDatasets(testX[0], testY)

    batch_size = 64
    train_size = len(trainDatasets)
    val_size = len(valDatasets)
    # test_size = test_data.test_data.shape[0]

    train_loader = torch.utils.data.DataLoader(trainDatasets,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               collate_fn=padding)

    val_loader = torch.utils.data.DataLoader(valDatasets,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             collate_fn=padding)

    test_loader = torch.utils.data.DataLoader(testDatasets,
                                              batch_size=1,
                                              shuffle=False)

    speechmodel = CTCSpeech()

    criterion = nn.CTCLoss()

    optimizer = optim.Adam(speechmodel.parameters(), lr=0.001)
    # optimizer = optim.SGD(speechmodel.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_ft = train_epoch(speechmodel, train_loader, val_loader, criterion, optimizer, 12, train_size, val_size)



if __name__ == '__main__':
    main()
