from music21 import converter, instrument, note, chord, stream, interval, pitch
import sys
import os
import pickle
from tqdm import tqdm

dir_ = 'data/all_key'
train_pickle_file = 'data/train_out.pkl'
test_pickle_file = 'data/test_out.pkl'
full_pickle_file = 'data/out.pkl'
def main():

    all_files = []
    all_melody = []
    all_chords = []
    sixteenth = 0.25 # 16th note: 1 = quarter note, 0.5 = 8th note
    CHORD_NAME = True # use chord name instead of chord notes

    for file in tqdm(os.listdir(dir_)):
        try:
            midi = converter.parseFile(dir_ + '/' + file)
        except:
            print('Could not parse ' + file)
            continue

        offset = 0
        stop = midi.highestTime
        song_melody = []
        song_chords = []

        while offset < stop:
            cur_melody = []
            cur_chords = []
            all_notes = midi.recurse().getElementsByOffsetInHierarchy(
                                        offset,
                                        offsetEnd=offset+sixteenth,
                                        mustBeginInSpan=False,
                                        includeElementsThatEndAtStart=False).notes

            # gather notes and cur_chords played at offset
            for element in all_notes:
                if isinstance(element, note.Note):
                    cur_melody.append(str(element.pitch.name))
                elif isinstance(element, chord.Chord):
                    if CHORD_NAME:
                        cur_chords.append(element.pitchedCommonName)
                    else:
                        cur_chords.append('.'.join(sorted([str(p.name) for p in element.pitches])))

            # nothing playing at offset
            if len(cur_melody) == 0 and len(cur_chords) == 0:
                song_melody.append('')
                song_chords.append('')

            # cur_melody played but not chord at offset
            elif len(cur_chords) == 0:
                for n in cur_melody:
                    song_melody.append(n)
                    song_chords.append('')

            # cur_chords played but not cur_melody at offset
            elif len(cur_melody) == 0:
                for c in cur_chords:
                    song_melody.append('')
                    song_chords.append(c)

            # both played at offset
            else:
                for n in cur_melody:
                    for c in cur_chords:
                        song_melody.append(n)
                        song_chords.append(c)

            offset += sixteenth

        all_files.append(file)
        all_melody.append(song_melody)
        all_chords.append(song_chords)

    # put into dictionary and send to pickle file
    harmony = {}
    harmony['file'] = all_files
    harmony['melody'] = all_melody
    harmony['chords'] = all_chords

    with open(full_pickle_file, 'wb') as filepath:
        pickle.dump(harmony, filepath)



# def split_dataset():
# 	import torch
# 	import pickle
# 	data_path = 'data/out.pkl'
# 	with open(data_path, 'rb') as f:
# 		full_dataset = pickle.load(f)
# 		train_size = int(0.8 * len(full_dataset))
# 		test_size = len(full_dataset) - train_size
# 		train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
# 		pickle.dump(train_dataset, open('./data/train_dataset.pkl', 'wb'))
# 		pickle.dump(test_dataset, open('./data/test_dataset.pkl', 'wb'))

def split(harmony):

    import random

    from sklearn.model_selection import train_test_split

    train_size = int(0.8 * len(harmony['file']))
    test_size = len(harmony['file']) - train_size

    x_train, x_test = train_test_split(list(range(len(harmony['file']))), test_size=0.2, random_state=1)

    # print(x_train)
    # print(x_test)
    train_harmony = {'file': [], 'melody': [], 'chords': []}
    test_harmony = {'file': [], 'melody': [], 'chords': []}
    for idx in x_train:
        train_harmony['file'].append(harmony['file'][idx])
        train_harmony['melody'].append(harmony['melody'][idx])
        train_harmony['chords'].append(harmony['chords'][idx])

    for idx in x_test:
        test_harmony['file'].append(harmony['file'][idx])
        test_harmony['melody'].append(harmony['melody'][idx])
        test_harmony['chords'].append(harmony['chords'][idx])

    print(train_harmony['file'][0])
    print(train_harmony['file'][1])
    print(train_harmony['melody'][0])
    print(train_harmony['melody'][1])
    print(train_harmony['chords'][0])
    print(train_harmony['chords'][1])
    with open(train_pickle_file, 'wb') as filepath:
        pickle.dump(train_harmony, filepath)
    with open(test_pickle_file, 'wb') as filepath:
        pickle.dump(test_harmony, filepath)



# main()

harmony = pickle.load(open(full_pickle_file, 'rb'))

split(harmony)


print('MIDI Processing Completed.')
