import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from base import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_loader.dataset import MidiDataset

"""
Model
"""
class BigMidiLSTM(MidiLSTM):
    """Bigger version of the Midi LSTM
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_network =  nn.Sequential()
        self.melody_classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * self.scale, self.hidden_size//2),
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
            nn.Linear(self.hidden_size * self.scale, self.hidden_size//2),
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

"""
Melody metrics
"""
def melody_preprocess(output, target, extra=None):
    '''
    Preprocessing for all melody metrics
    Returns
        flat_melody_out (batch_size * seq_len, vocab_size)
        flat_melody_y (batch_size * seq_len)

        EDIT_R - Took out flat_melody_out and _y cuz they're not necessary at this point
    '''
    melody_out = torch.exp(output['melody_out'].detach())
    melody_y = target['melody_y'].detach()

    return melody_out, melody_y

def melody_accuracy(output, target, extra=None):
    melody_out, melody_y = melody_preprocess(output, target, extra=extra)
    return accuracy(melody_out, melody_y)

def melody_accuracy_topk(output, target, extra=None):
    melody_out, melody_y = melody_preprocess(output, target, extra=extra)
    return accuracy_topk(melody_out, melody_y)

"""
Chord metrics
"""
def chord_preprocess(output, target, extra=None):
    '''
    Preprocessing for all melody metrics
    Returns
        flat_chord_out (batch_size * seq_len * vocab_size)
        flat_chord_y (batch_size * seq_len * vocab_size)
    '''
    seq_lengths = extra['seq_lengths']
    chord_out = output['chord_out'].detach()
    chord_y = target['chord_y'].detach()

    batch_size = len(seq_lengths)

    flat_chord_out = chord_out
    flat_chord_y = chord_y
    
    flat_chord_out = torch.exp(flat_chord_out)
    flat_chord_y = flat_chord_y.long()

    return flat_chord_out, flat_chord_y

def chord_accuracy(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return accuracy(chord_out, chord_y)

def chord_accuracy_topk(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return accuracy_topk(chord_out, chord_y)

def chord_multilabel_accuracy(output, target, extra=None, thresh=0.5):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return multilabel_accuracy(chord_out, chord_y, thresh=thresh)

def chord_precision(output, target, extra=None, thresh=0.5):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return metrics.precision_score(chord_y, chord_out > thresh)

def chord_recall(output, target, extra=None, thresh=0.5):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return metrics.recall_score(chord_y, chord_out > thresh)

def chord_average_precision(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return metrics.average_precision_score(chord_y, chord_out)

def chord_auc(output, target, extra=None):
    chord_out, chord_y = chord_preprocess(output, target, extra=extra)
    return metrics.roc_auc_score(chord_y, chord_out)

"""
Generic metrics
"""
def multilabel_accuracy(output, target, thresh=0.5):
    """
    output: torch.tensor
        (batch_size,)
    target: torch.tensor
        (batch_size,)
    """
    with torch.no_grad():
        pred = (output > 0.5).float()
        assert pred.shape == target.shape
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy(output, target):
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

"""
Loss
"""
def midi_loss(output, target, extra=None):
    """Loss for the midi lstm model
    Has loss for both the chord and the melody
    """
    melody_out = output['melody_out']
    chord_out = output['chord_out']
    melody_y = target['melody_y']
    chord_y = target['chord_y']

    batch_size = melody_out.size(0)
    seq_lengths = extra['seq_lengths']
    device = melody_out.device

    flat_melody_out = melody_out
    flat_chord_out = chord_out
    flat_melody_y = melody_y
    flat_chord_y = chord_y

    # melody_out = melody_out[:, seq_len-1].contiguous().view(-1, vocab_size)
    # chord_out = chord_out[:, seq_len-1].contiguous().view(-1, vocab_size)
    # melody_y = melody_y[:, seq_len-1].flatten()
    # chord_y = chord_y[:, seq_len-1].view(-1, vocab_size)

    chord_loss = F.nll_loss
    flat_chord_y = flat_chord_y.long()

    return F.nll_loss(flat_melody_out, flat_melody_y) + chord_loss(flat_chord_out, flat_chord_y)

def midi_loss_chord_only(output, target, extra=None):
    """Midi model loss for the chord only (no melody)
    """
    melody_out = output['melody_out']
    chord_out = output['chord_out']
    melody_y = target['melody_y']
    chord_y = target['chord_y']

    batch_size = melody_out.size(0)
    seq_lengths = extra['seq_lengths']
    device = melody_out.device

    flat_melody_out = melody_out
    flat_chord_out = chord_out
    flat_melody_y = melody_y
    flat_chord_y = chord_y

    # melody_out = melody_out[:, seq_len-1].contiguous().view(-1, vocab_size)
    # chord_out = chord_out[:, seq_len-1].contiguous().view(-1, vocab_size)
    # melody_y = melody_y[:, seq_len-1].flatten()
    # chord_y = chord_y[:, seq_len-1].view(-1, vocab_size)

    chord_loss = F.nll_loss
    flat_chord_y = flat_chord_y.long()

    return chord_loss(flat_chord_out, flat_chord_y)
