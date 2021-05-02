import torch

from utils.util import get_instance
import model.model as module_model
from data_loader.dataset import MidiDataset
from utils.util import convert_chord_int_to_str, convert_note_to_int

class Predict:
    def __init__(self, data_path='./data/out.pkl', resume='./weights/Music_LSTM_big/0304_041925/model_best.pth'):
        """
        Initialize the predict class
        """
        self.dataset = MidiDataset(data_path)

        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(resume)
        else:
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        
        # Load model
        self.model = get_instance(module_model, 'model', checkpoint['config'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()


    def generateOutput(self, input, extra=None):
        """
        Parameters
        ----------
        input: dict
            input into the model
        extra: dict
            extra information for the input

        Returns
        -------
        pred_ouput: list
            list of the chords to return
        """
        pred_output = []
        with torch.no_grad():
            output = self.model(input, extra=extra)

        output_chords = output['chord_out'].squeeze()

        for chord in output_chords:
            c_idx = int(torch.argmax(chord))
            chordstr = convert_chord_int_to_str(c_idx)
            pred_output.append(chordstr)

        return pred_output


    def process(self, melody):
        """ Preprocess the melody for prediction
        Parameters
        ----------
        melody: list
            list of notes

        Returns
        -------
        Input: dict
            dict of inputs to model
        Extra: dict
            dict of extra inputs to model
        """
        x = [convert_note_to_int(n) for n in melody]
        seq_lengths = [len(x)]

        # convert to torch tensor
        x = torch.LongTensor(x).unsqueeze(0)
        seq_lengths = torch.LongTensor(seq_lengths)

        # store in output
        input = {'melody_x':x}
        extra = {'seq_lengths': seq_lengths}
        return input, extra

    def get_prediction(self, input_):
        """
        Parameters
        ----------
        input_: list of strings
            List of notes

        Returns
        -------
        list of chords
            list of chords to play
        """
        x, extra = self.process(input_)
        return self.generateOutput(x, extra=extra)

if __name__ == "__main__":
    predict = Predict()
    input = ['A','A','A','A','A','A','C','C','C','C','C','C','C','C','C','C','A','A','B','B','B','B','B','A']
    # inputs = [input, input]
    output = predict.get_prediction(input)
    print(input)
    print(output)
