from utils.util import convert_chord_to_onehot, convert_note_to_int, \
                        NOTES_TO_INT, INT_TO_NOTES, CHORD_TO_INT, INT_TO_CHORD, CHORD_TO_NOTES, NOTES_TO_CHORD
import os
import pandas as pd
import pickle
from torch.utils.data import Dataset

class MidiDataset(Dataset):
    """MIDI Music Dataset"""

    NUM_NOTES = max(NOTES_TO_INT.values()) + 1
    NOTES_TO_INT = NOTES_TO_INT
    INT_TO_NOTES = INT_TO_NOTES
    CHORD_TO_INT = CHORD_TO_INT
    INT_TO_CHORD = INT_TO_CHORD
    CHORD_TO_NOTES = CHORD_TO_NOTES
    NOTES_TO_CHORD = NOTES_TO_CHORD


    NUM_CHORDS = max(CHORD_TO_INT.values()) + 1

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

        notes = [convert_note_to_int(n) for n in row['melody']]
        melody_x = notes[:-1] # all but last note
        melody_y = notes[1:] # all but first note
        chord_y = [convert_chord_to_onehot(c) for c in row['chords']][:-1] # all but last

        return melody_x, melody_y, chord_y

if __name__ == '__main__':
    data_path = 'data/out.pkl'
    dataset = MidiDataset(data_path)
