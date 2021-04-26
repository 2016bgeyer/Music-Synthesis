import numpy as np
from midiutil import MIDIFile
import pdb
import random
import math

resolution = 2400.0
    
def add_row(matrix, row):
    if(len(matrix) == 0):
        return np.array([row])
    else:
        return np.vstack([matrix, row])

def note_on(arr, note_data):
    start = note_data[1] / resolution
    stop = note_data[2] / resolution
    pitch = note_data[3]    
    shape = arr.shape
    w_shape = shape
    if(shape[0] < stop):
        w_shape = (stop, 128)
    working = arr.flatten()
    working.resize(w_shape)
    '''
    working = np.zeros(w_shape)
    if(len(arr) > 0):
        print(working.shape, arr.shape, shape[0])
        working[0:shape[0]][0:shape[1]] = arr #errors for some reason
    '''
#    working = np.copy(arr)
#    if(shape[0] < stop):
#        working.resize((stop, 128))
    #np.put replaces on a flattened array
    np.put(working, range((start * 128) + pitch, ((stop - 1) * 128) + pitch + 1, 128), ([ 1 ]))
    # print(working)
    return working

def validate_data(data):
# correct data format is a 2d numpy array data[i][j], where i represents the note in the sequence
# j values:
#   0: composition number (should be the same for all i in a data instance)
#   1: start time
#   2: end time
#   3: midi note number/pitch
#   4: part number (i.e. channel)
    return 0

#reads a csv file and returns a 3d array data[i][j][k]
#each i is a separate piece of music
#each j is a note in that piece
#k are the variables associated with each note
#max_data is the maximum number of pieces to read from the csv
def csv_to_data(filename, max_data):
    with open(filename, "r") as file:
        output = np.empty(max_data, dtype=object)
        file.readline() #csv header
        data = np.array([], dtype=int)
        i = 0
        while i < max_data:
            line = file.readline()
            if(line == ""): #eof
                if(i < (max_data - 1)):
                    output = np.resize(output, i + 1)
                return output
            note = [int(j, base=10) for j in line.split(",")]
            if(i == note[0]):
#                data = note_on(data, note)
                if(len(data) == 0):
                    data = np.array([[note[1], note[2], note[3]]], dtype=int)
                else:
                    data = np.vstack((data, np.array([ note[1], note[2], note[3] ])))
            else:
                output[i] = data
                i += 1
                data = np.array([[note[1], note[2], note[3]]], dtype=int)
#                data = note_on(data, note)
    return output
    
#used to create artificial data (labeled as artificial) for the discriminator
def create_false_data(amount=1):
    output = np.empty(amount, dtype=object)
    for i in range(amount):
        length = 170 + int(random.random() * 10)
        data = np.array([], dtype=int)
        for note in range(length):
            start = int(random.random() * 200000)
            stop = int(random.random() * 200000)
            pitch = int(random.random() * 127)
            if(start > stop):
                swap = start
                start = stop
                stop = start
            if(note == 0):
                data = np.array([[ start, stop, pitch ]], dtype=int)
            else:
                data = np.vstack((data, np.array([start, stop, pitch], dtype=int)))
        output[i] = data
    return output
    
#takes one data instance (a 2d numpy array) and writes it to a midi file
def write_midi(data, filename):
    midi = MIDIFile(numTracks=1, deinterleave=False)
    midi.addTempo(0, 0, 120)
    limit = 16 # limit to nearest Xnd note
    for note in data:
        start_n = float(note[0] * 190.8)
        stop_n = float(note[1] * 190.8) + start_n
        start = round(limit * start_n) / limit
        stop = round(limit * stop_n) / limit
        if(start >= stop):
            continue
        pitch = int(note[2] * 51 + 45)
#        print(pitch, start, stop)
        midi.addNote(0, 0, pitch, start, stop - start, 100)
    '''
    for pitch in range(data.shape[1]):
        pitch_notes = data[:, pitch]
        on = False
        start = None
        for i in range(len(pitch_notes)):
            if(on is False and pitch_notes[i] == 1):
                on = True
                start = i * 10
            if(on is True and pitch_notes[i] == 0):
                on = False
                stop = i * 10
                midi.addNote(0, 0, pitch, start / resolution, (stop - start) / resolution, 100)
    '''
#    for note in data:
#        midi.addNote(0, 0, note[3], note[1] / resolution, (note[2] - note[1]) / resolution, 100)
    with open(filename, "wb") as file:
        midi.writeFile(file)
    return filename

def write_training_data():
    midi = MIDIFile(numTracks=1, deinterleave=False)
    midi.addTempo(0, 0, 120)
    open_data = open("jsb_train.csv")

    for line in open_data:
        a = line.split(",")
        if a[1] != "t0" and a[0] == "0":
            a = a[0:4]
            midi.addNote(0, 0, int(a[3]), int(a[1]) / resolution, (int(a[2]) - int(a[1])) / resolution, 100)
    with open("train_file.mid", "wb") as file:
        midi.writeFile(file)        