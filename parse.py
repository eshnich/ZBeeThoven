import tensorflow as tf
import numpy as np
import music21 as m21

test_midi = 'http://kern.ccarh.org/cgi-bin/ksdata?l=cc/bach/cello&file=bwv1007-01.krn&f=xml'

def parse_music(file_name, num_voices=1, show=False):
    # returns an array of [note, duration] elements
    #   - note is a string (ex. 'A4')
    #   - duration is a float representing number of quarter notes (ex. 0.25)

    piece = m21.converter.parse(file_name)

    if show:
        piece.show()

    if num_voices == 1:

        metadata = piece[0]
        stream = piece[1]
        instrument = stream[0]
        measures = stream[1:]
        data = []

        for measure in measures:
            for note in measure.notes:
                try:
                    data.append([note.nameWithOctave, note.duration.quarterLength]) # type: [string, float]
                except:
                    pass
        return data
        
    return

print(parse_music(test_midi, show=False))