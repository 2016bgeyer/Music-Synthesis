import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .constants import NOTES_TO_INT, INT_TO_NOTES, CHORD_TO_INT, INT_TO_CHORD, CHORD_TO_NOTES, NOTES_TO_CHORD


'''
TODO LOOK AT THIS
https://www.reddit.com/r/musictheory/comments/1jd894/looking_for_an_algorithm_that_generates_chord/
https://www.reddit.com/r/cs231n/comments/93iyxa/cs231nrepodeepubuntutargz_not_found/
'''
class MidiDataset(Dataset):
    """MIDI Music Dataset"""
    # NOTES = ['C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'E#', 'F-', 'F', 'F#',
    #             'G-', 'G', 'G#', 'A-', 'A', 'A#', 'B-', 'B', 'B#', 'C-']
    # NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # NOTE_TO_INT = {k:v for v, k in enumerate(NOTES)}

    NUM_NOTES = max(NOTES_TO_INT.values()) + 1
    NOTES_TO_INT = NOTES_TO_INT
    INT_TO_NOTES = INT_TO_NOTES
    CHORD_TO_INT = CHORD_TO_INT
    INT_TO_CHORD = INT_TO_CHORD
    CHORD_TO_NOTES = CHORD_TO_NOTES
    NOTES_TO_CHORD = NOTES_TO_CHORD


    NUM_CHORDS = max(CHORD_TO_INT.values()) + 1
    INTERSECT_THRESH = 2 # Number of notes to intersect to count as a chord

    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to pickle file containing the dictionary of data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        assert os.path.exists(data_path), "{} does not exist".format(data_path)
        if not data_path.endswith('.pkl'):
            raise IOError('{} is not a recoginizable file type (.pkl)'.format(data_path))

        # load data
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.df = pd.DataFrame.from_dict(data_dict)

    @classmethod
    def convert_note_to_int(cls, note):
        return cls.NOTES_TO_INT[note]

    @classmethod
    def convert_chord_to_int(cls, chord):
        """
        converts chord into list of ints
        Parameters
        ----------
        chord : str
            notes in chord
        Returns
        -------
        note_list : list
            list of ints for each note in chord
        """
        note_list = []
        notes = chord.split('.')

        for n in notes:
            note_list.append(cls.convert_note_to_int(n))

        return note_list

    def convert_chord_to_onehot(self, chord):
        """"Convert chord to onehot list"""
        chord_name = 'EMPTY'
        for k,v in NOTES_TO_CHORD.items():
            if len(set(chord.split('.')).intersection(set(k.split('.')))) >= self.INTERSECT_THRESH:
                chord_name = v
                break

        # offset by 1 to remove pad
        return self.CHORD_TO_INT[chord_name]

    @classmethod
    def convert_chord_to_binary(cls, chord):
        """Convert chord to binary list"""
        note_list = [0] * cls.NUM_NOTES
        notes = chord.split('.')

        for n in notes:
            note_list[cls.NOTES_TO_INT[n]] = 1

        return note_list

    @classmethod
    def convert_int_to_note(cls, int):
        """Convert int to the note"""
        return cls.INT_TO_NOTES[int]

    @classmethod
    def convert_binary_to_chord(cls, binary_list):
        """Convert binary list to the chord"""
        notes = np.argmax(binary_list)
        chord = '.'.join(convert_int_to_note(n) for n in notes)

        return chord

    @classmethod
    def convert_chord_int_to_str(cls, chord_int):
        """"Convert chord to string"""
        return cls.CHORD_TO_NOTES[cls.INT_TO_CHORD[chord_int]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): idx of data entry to get
        Returns:
            sample : melody_x, melody_y, chord_y
        """
        row = self.df.iloc[idx]

        notes = [self.convert_note_to_int(n) for n in row['melody']]
        melody_x = notes[:-1] # all but last note
        melody_y = notes[1:] # all but first note
        chord_y = [self.convert_chord_to_onehot(c) for c in row['chords']][:-1] # all but last

        return melody_x, melody_y, chord_y

if __name__ == '__main__':
    data_path = 'data/out.pkl'
    dataset = MidiDataset(data_path)
