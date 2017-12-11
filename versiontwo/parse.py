import tensorflow as tf
import numpy as np
import music21 as m21
import collections

test_midi = 'http://kern.ccarh.org/cgi-bin/ksdata?l=cc/bach/cello&file=bwv1007-01.krn&f=xml'

def build_dataset(notes):
    # notes is an array of (note, duration) elements
    #   - note is a string (ex. 'A4')
    #   - duration is a float representing number of quarter notes (ex. 0.25)
    count = collections.Counter(notes).most_common()
    dictionary = {}
    for note, _ in count:
        dictionary[note] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def parse_music(file_name, voices=1, show=False):
    # returns an array of (note, duration) elements
    #   - note is a string (ex. 'A4')
    #   - duration is a float representing number of quarter notes (ex. 0.25)
    piece = m21.converter.parse(file_name)

    if show:
        piece.show()

    if voices == 1:
        data = []
        for note in piece.flat.elements:
            try:
                data.append((note.nameWithOctave, note.duration.quarterLength))
            except:
                pass
        return data

    return

    # metadata = piece[0]
    # print('metadata: {}'.format(metadata))
    # # stream = piece[1]
    # # instrument = stream[0]
    # # measures = stream[1:]
    # for voice in metadata.voices:
    #     data = []
    #     for note in voice.notes.stream():
    #         try:
    #             data.append((note.nameWithOctave, note.duration.quarterLength)) # type: [string, float]
    #         except:
    #             pass
    #     test_data.append(data)
    #     dictionary_data.extend(data)
    # print('test data: {}'.format(test_data))
    # print('dictionary data: {}'.format(dictionary_data))
    # return test_data, dictionary_data

# vec_to_num, num_to_vec = build_dataset(parse_music(test_midi))
# print(vec_to_num)
# print(num_to_vec)













