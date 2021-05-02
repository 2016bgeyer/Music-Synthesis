import os

'''
Constants for the dataset
'''

# NOTES = ['C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'E#', 'F-', 'F', 'F#',
#             'G-', 'G', 'G#', 'A-', 'A', 'A#', 'B-', 'B', 'B#', 'C-']
# NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# NOTE_TO_INT = {k:v for v, k in enumerate(NOTES)}
NOTES_TO_INT = {
    'PAD':0, # padding
    '':1, # empty note
    'A':2,
    'A#':3,     'B-':3,
    'B':4,      'C-':4,
    'B#':5,     'C':5,
    'C#':6,     'D-':6,
    'D':7,
    'D#':8,     'E-':8,
    'E':9,      'F-':9,
    'E#':10,    'F':10,
    'F#':11,    'G-':11,
    'G':12,
    'G#':13,    'A-':13,
}

INT_TO_NOTES = {v:k for k,v in NOTES_TO_INT.items()}

CHORD_TO_INT = {
    'PAD':0,
    'EMPTY':1,
    'A':2,
    'Am':3,
    'A#':4,     'B-':4,
    'A#m':5,    'B-m':5,
    'B':6,      'C-':6,
    'Bm':7,     'C-m':7,
    'B#':8,     'C':8,
    'B#m':9,    'Cm':9,
    'C#':10,    'D-':10,
    'C#m':11,   'D-m':11,
    'D':12,
    'Dm':13,
    'D#':14,    'E-':14,
    'D#m':15,   'E-m':15,
    'E':16,     'F-':16,
    'Em':17,    'F-m':17,
    'E#':18,    'F':18,
    'E#m':19,   'Fm':19,
    'F#':20,    'G-':20,
    'F#m':21,   'G-m':21,
    'G':22,
    'Gm':23,
    'G#':24,    'A-':24,
    'G#m':25,   'A-m':25,
}

INT_TO_CHORD = {v:k for k,v in CHORD_TO_INT.items()}


CHORD_TO_NOTES = {
    'PAD':'',
    'EMPTY':'',
    'A':'A.C#.E',
    'Am':'A.C.E',
    'A#':'A#.C#.E#',    'B-':'B-.D-.F',
    'A#m':'A#.C.E#',    'B-m':'B-.C.F',
    'B':'B.D#.F#',      'C-':'C-.E-.G-',
    'Bm':'B.D.F#',      'C-m':'C-.D.G-',
    'B#':'B#.E.G',      'C':'C.E.G',
    'B#m':'B#.E-.G',    'Cm':'C.E-.G',
    'C#':'C#.E#.G#',    'D-':'D-.F.A-',
    'C#m':'C#.E.G#',    'D-m':'D-.F-.A-',
    'D':'D.F#.A',
    'Dm':'D.F.A',
    'D#':'D#.G.A#',     'E-':'E-.G.B-',
    'D#m':'D#.G-.A#',   'E-m':'E-.G-.B-',
    'E':'E.G#.B',       'F-':'F-.A-.B',
    'Em':'E.G.B',       'F-m':'F-.G.B',
    'E#':'E#.A.B#',     'F':'F.A.C',
    'E#m':'E#.G#.B#',   'Fm':'F.A-.C',
    'F#':'F#.A#.C#',    'G-':'G-.B-.D-',
    'F#m':'F#.A.C#',    'G-m':'G-.A.D-',
    'G':'G.B.D',
    'Gm':'G.B-.D',
    'G#':'G#.B#.D#',    'A-':'A-.C-.E-',
    'G#m':'G#.B.D#',    'A-m':'A-.B.E-',
}

INTERSECT_THRESH = 2 # Number of notes to intersect to count as a chord


# SORT NOTES IN CHORDS ALPHABETICALLY
for k, v in CHORD_TO_NOTES.items():
    CHORD_TO_NOTES[k] = '.'.join(sorted(v.split('.')))

NOTES_TO_CHORD = {v:k for k,v in CHORD_TO_NOTES.items()}


def get_instance(module, name, config, *args, **kwargs):
    """Get instance from a certain module using the args and kwargs
    """
    print(name)
    return getattr(module, config[name]['name'])(*args, **kwargs, **config[name]['args'])

def convert_chord_int_to_str(chord_int):
    """"Convert chord to string"""
    return CHORD_TO_NOTES[INT_TO_CHORD[chord_int]]

def convert_chord_to_onehot(chord):
    """"Convert chord to onehot list"""
    chord_name = 'EMPTY'
    for k,v in NOTES_TO_CHORD.items():
        if len(set(chord.split('.')).intersection(set(k.split('.')))) >= INTERSECT_THRESH:
            chord_name = v
            break

    # offset by 1 to remove pad
    return CHORD_TO_INT[chord_name]

def convert_note_to_int(note):
    return NOTES_TO_INT[note]

def convert_chord_to_int(chord):
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
        note_list.append(convert_note_to_int(n))

    return note_list