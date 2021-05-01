import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_loader.dataset import MidiDataset

'''
Example models
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
https://github.com/claravania/lstm-pytorch/blob/master/model.py
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
https://github.com/warmspringwinds/pytorch-rnn-sequence-generation-classification/blob/master/notebooks/music_generation_training_nottingham.ipynb
'''

class MidiLSTM(BaseModel):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        """
        Parameters
        ----------
        vocab_size : int
            Dimension of vocabulary
        hidden_size : int
            Dimension of the hidden layer outputs
        num_layers : int
            Number of LSTMs stacked on each other
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.scale = 2 if self.bidirectional else 1

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.hidden_network = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * self.scale, self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.melody_classifier = nn.Linear(self.hidden_size//8, self.vocab_size)
        if MidiDataset.USE_CHORD_ONEHOT:
            self.chord_classifier = nn.Linear(self.hidden_size//8, MidiDataset.NUM_CHORDS)
        else:
            self.chord_classifier = nn.Linear(self.hidden_size//8, self.vocab_size)

    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden input for LSTM
            (num_layers, batch_size, hidden_size)
        """
        return (torch.zeros(self.num_layers * self.scale, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers * self.scale, batch_size, self.hidden_size).to(device))

    def forward(self, data, extra=None):
        """
        Forward pass through LSTM
        Parameters
        ----------
        data:
            melody_x : torch.tensor (batch_size, sequence_len)
                input to model
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
        if MidiDataset.USE_CHORD_ONEHOT:
            chord_out = F.log_softmax(self.chord_classifier(out), dim=2)
        else:
            chord_out = torch.sigmoid(self.chord_classifier(out))

        # transpose from (seq_len, batch_size, ...) -> (batch_size, seq_len, ...)
        melody_out = melody_out.transpose(0, 1)
        chord_out = chord_out.transpose(0, 1)

        if MidiDataset.USE_SEQUENCE: # always False
            output = {
                'melody_out': melody_out[:, -1],
                'chord_out': chord_out[:, -1]
            }
        else:
            output = {
                'melody_out': melody_out,
                'chord_out': chord_out
            }

        return output


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
            nn.Linear(self.hidden_size//8, self.vocab_size),
        )
        if MidiDataset.USE_CHORD_ONEHOT:    # always true
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
                nn.Linear(self.hidden_size//8, MidiDataset.NUM_CHORDS),
            )
        else:
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
                nn.Linear(self.hidden_size//8, self.vocab_size),
            )