import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
import logging
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_loader.dataset import MidiDataset

"""
Model
"""
class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(MyModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.hidden_network =  nn.Sequential()
        self.melody_classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//8, self.vocab_size)
        )
        self.chord_classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//8, MidiDataset.NUM_CHORDS)
        )

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(MyModel, self).__str__() + '\nTrainable parameters: {}'.format(params)

    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden input for LSTM
            (num_layers, batch_size, hidden_size)
        """
        return (torch.zeros(self.num_layers * (int(self.bidirectional) + 1), batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers * (int(self.bidirectional) + 1), batch_size, self.hidden_size).to(device))

    def forward(self, data, extra=None):
        """
        Forward pass through LSTM
        Parameters
        ----------
        data:
            melody_x : torch.tensor (batch_size, sequence_len)
                input to model
        extra:
            seq_lengths for use in packing/padding

        Returns
        -------
        output:
            melody_out : torch.tensor (batch_size, sequence_len, vocab_size)
                Melody softmax output (only want one note)
            chord_out : torch.tensor (batch_size, sequence_len, vocab_size)
                Chord sigmoid output (want chords)
        """
        x = data['melody_x']
        seq_lengths = extra['seq_lengths']
        batch_size = x.size(0)

        # transpose from (batch_size, seq_len, ...) -> (seq_len, batch_size, ...)
        x = x.transpose(0, 1)

        self.hidden = self.init_hidden(batch_size, device=x.device)
        embed = self.embed(x)

        # pack up sequences by length
        packed_embed = pack_padded_sequence(embed, seq_lengths)
        out, self.hidden = self.lstm(packed_embed, self.hidden)
        out, out_lengths = pad_packed_sequence(out)

        out = self.hidden_network(out)

        melody_out = F.log_softmax(self.melody_classifier(out), dim=2)
        chord_out = F.log_softmax(self.chord_classifier(out), dim=2)

        # transpose from (seq_len, batch_size, ...) -> (batch_size, seq_len, ...)
        melody_out = melody_out.transpose(0, 1)
        chord_out = chord_out.transpose(0, 1)


        return {'melody_out': melody_out,'chord_out': chord_out}

def melody_preprocess(output, target, extra):
    '''
    Preprocessing for all melody metrics
    Returns
        flat_melody_out (batch_size * seq_len, vocab_size)
        flat_melody_y (batch_size * seq_len)
    '''
    melody_out = torch.exp(output['melody_out'].detach()) # log to normal
    melody_y = target['melody_y'].detach()

    seq_lengths = extra['seq_lengths']
    batch_size = len(seq_lengths)

    flat_melody_out = torch.tensor([])
    flat_melody_y = torch.tensor([]).long() # try moving both .long()s to after the loop
    for i in range(batch_size):
        flat_melody_out = torch.cat((flat_melody_out, melody_out[i, :seq_lengths[i]]))
        flat_melody_y = torch.cat((flat_melody_y, melody_y[i, :seq_lengths[i]].long()))

    return flat_melody_out, flat_melody_y

def melody_accuracy(output, target, extra=None):
    melody_out, melody_y = melody_preprocess(output, target, extra)
    return accuracy(melody_out, melody_y)

def melody_accuracy_topk(output, target, extra=None):
    melody_out, melody_y = melody_preprocess(output, target, extra)
    return accuracy_topk(melody_out, melody_y)

def chord_preprocess(output, target, extra=None):
    chord_out = output['chord_out'].detach()
    chord_y = target['chord_y'].detach()

    seq_lengths = extra['seq_lengths']
    batch_size = len(seq_lengths)

    flat_chord_out = torch.tensor([])
    flat_chord_y = torch.tensor([])
    for i in range(batch_size):
        flat_chord_out = torch.cat((flat_chord_out, chord_out[i, :seq_lengths[i]]))
        flat_chord_y = torch.cat((flat_chord_y, chord_y[i, :seq_lengths[i]].float()))
    
    flat_chord_out = torch.exp(flat_chord_out) # log to normal
    flat_chord_y = flat_chord_y.long()

    return flat_chord_out, flat_chord_y

def chord_accuracy(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return accuracy(chord_out, chord_y)

def chord_accuracy_topk(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return accuracy_topk(chord_out, chord_y)

def chord_multilabel_accuracy(output, target, extra=None, thresh=0.5):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return multilabel_accuracy(chord_out, chord_y, thresh=thresh)

def chord_precision(output, target, extra=None, thresh=0.5):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return metrics.precision_score(chord_y, chord_out > thresh)

def chord_recall(output, target, extra=None, thresh=0.5):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return metrics.recall_score(chord_y, chord_out > thresh)

def chord_average_precision(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return metrics.average_precision_score(chord_y, chord_out)

def chord_auc(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra)
    return metrics.roc_auc_score(chord_y, chord_out)

def multilabel_accuracy(output, target, extra=None, thresh=0.5):
    """
    output: torch.tensor
        (batch_size,)
    target: torch.tensor
        (batch_size,)
    """
    with torch.no_grad():
        pred = (output > thresh).float()
        assert pred.shape == target.shape
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy(output, target, extra=None):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_topk(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def chord_and_melody_loss(output, target, extra=None):
    return melody_loss(output, target, extra) + chord_loss(output, target, extra)

def melody_loss(output, target, extra=None):
    melody_out = output['melody_out']
    melody_y = target['melody_y']

    batch_size = melody_out.size(0)
    seq_lengths = extra['seq_lengths']
    device = melody_out.device

    # flatten (batchsize, seq_len, vocab_size) -> (batchsize * seq_len, vocab_size)
    flat_melody_out = torch.tensor([]).to(device)
    flat_melody_y = torch.tensor([]).long().to(device)
    for i in range(batch_size):
        flat_melody_out = torch.cat((flat_melody_out, melody_out[i, :seq_lengths[i]]))
        flat_melody_y = torch.cat((flat_melody_y, melody_y[i, :seq_lengths[i]].long()))

    return F.nll_loss(flat_melody_out, flat_melody_y)

def chord_loss(output, target, extra=None):
    chord_out = output['chord_out']
    chord_y = target['chord_y']

    batch_size = output['melody_out'].size(0)
    device = output['melody_out'].device
    seq_lengths = extra['seq_lengths']

    # flatten (batchsize, seq_len, vocab_size) -> (batchsize * seq_len, vocab_size)
    flat_chord_out = torch.tensor([]).to(device)
    flat_chord_y = torch.tensor([]).to(device)
    for i in range(batch_size):
        flat_chord_out = torch.cat((flat_chord_out, chord_out[i, :seq_lengths[i]]))
        flat_chord_y = torch.cat((flat_chord_y, chord_y[i, :seq_lengths[i]].float()))

    return F.nll_loss(flat_chord_out, flat_chord_y.long())
