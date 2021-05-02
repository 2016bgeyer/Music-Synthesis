import torch

def midi_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (melody_x, melody_y, chord_y).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (melody_x, melody_y, chord_y).
            - melody_x: torch tensor of shape (sequence_len,)
            - melody_y: torch tensor (sequence_len,)
            - chord_y: torch tensor (sequence_len, vocab_size) vocab_size = MidiDataset.NUM_NOTES
    Returns:
        data:
            - melody_x: torch tensor of shape (batch_size, sequence_len)
        target:
            - melody_y: torch tensor of shape (batch_size, sequence_len)
            - chord_y_onehot: torch tensor of shape (batch_size, sequence_len, vocab_size)
        extra:
            - seq_lengths: torch tensor of sequence lengths of batch
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)

    melody_x, melody_y, chord_y = zip(*data)
    batch_size = len(melody_x)
    
    # get length of each sequence in batch
    seq_lengths = torch.LongTensor([len(seq) for seq in melody_x])
    seq_melody_x = torch.zeros((batch_size, seq_lengths.max())).long()
    seq_melody_y = torch.zeros((batch_size, seq_lengths.max())).long()
    
    # make onehot chord tensor
    seq_chord_y = torch.zeros(batch_size, seq_lengths.max())
    for idx, (chord, seq_len) in enumerate(zip(chord_y, seq_lengths)):
        seq_chord_y[idx, :seq_len] = torch.tensor(chord)

    # pad to max sequence length
    for idx, (mx, my, seq_len) in enumerate(zip(melody_x, melody_y, seq_lengths)):
        seq_melody_x[idx, :seq_len] = torch.tensor(mx)
        seq_melody_y[idx, :seq_len] = torch.tensor(my)


    # sort tensors by length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_melody_x = seq_melody_x[perm_idx]
    seq_melody_y = seq_melody_y[perm_idx]
    seq_chord_y = seq_chord_y[perm_idx]

    # package in dictionary
    data = {'melody_x': seq_melody_x}
    target = {'melody_y': seq_melody_y, 'chord_y': seq_chord_y}
    extra = {'seq_lengths': seq_lengths}

    return data, target, extra
