import tensorflow as tf
import numpy as np
import music21 as m21

midi = 'http://kern.ccarh.org/cgi-bin/ksdata?l=cc/bach/cello&file=bwv1007-01.krn&f=xml'

piece = m21.converter.parse(midi)
piece.show()

metadata = piece[0]
stream = piece[1]
instrument = stream[0]
measures = stream[1:]

for m in measures:
	print(m)
	print(m.pitches)



#HELP
