import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils

from music21 import converter, instrument, note,chord, stream

from keras.utils.vis_utils import model_to_dot, plot_model
pd.set_option('display.max_columns', 999)




midi_list = ['All Too Well.mid','All the Small Things.mid', 'Beauty and the Beat.mid','Breakaway.mid','Cant Feel My Face.mid','Dont Matter.mid','Dont Stop Believing.mid','Falling Slowly.mid', 'Hallelujah.mid','Halo.mid','Im Yours.mid','Imagine.mid','Let It Be.mid','Purpose.mid','Somewhere Out There.mid','Stay With Me.mid','Stronger.mid','Sunday Morning.mid','We Belong Together.mid','When I was Your Man.mid']


def run(midi_list):

    all_midis = []
    all_parts = []
    
    for song in midi_list:
        midi = converter.parse('/var/www/lstm/LSTM music composer/midi_songs/' + song)
        for i in midi.parts:
            i.insert(0, instrument.Piano())
        parts = instrument.partitionByInstrument(midi)

        all_midis.append(midi)
        all_parts.append(parts)


    notes = []
    notes_offset = []
    durations = []

    for parts in all_midis:
        for i in parts.recurse():
            if isinstance(i, note.Note):
                notes.append(str(i.pitch))
                notes_offset.append(float(i.offset))
                durations.append(float(i.duration.quarterLength))

            elif isinstance(i, chord.Chord):
                notes_offset.append(float(i.offset))
                durations.append(float(i.duration.quarterLength))
                
                i = str(i).replace('>', '')
                chords = '|'.join(i.split()[1:])
                notes.append(chords)
            

    sequence_length = 100

    allnotes = sorted(set(i for i in notes))

    note_index_dic = dict((note, index) for index, note in enumerate(allnotes))

    notes_input = []
    notes_output = []
    n_words = len(set(notes))

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]

        notes_input.append([note_index_dic[note] for note in sequence_in])
        notes_output.append(note_index_dic[sequence_out])

    notes_input = np.array(notes_input).reshape(len(notes_input), sequence_length, 1)
    notes_output = np_utils.to_categorical(notes_output)


    notes_input = notes_input / float(n_words)

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(notes_input.shape[1], notes_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_words))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(notes_input, notes_output, epochs = 10, batch_size = 64)

    start = np.random.randint(0, len(notes_input))

    int_to_note = dict((index, note) for index, note in enumerate(allnotes))

    start_music = notes_input[start]


    prediction_output = []

    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(start_music, (1, len(start_music), 1))
        prediction_input = prediction_input / float(n_words)
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]

        prediction_output.append(result)

        start_music = list(start_music)
        start_music.append(index)
        start_music = start_music[1: len(start_music)]


    offset = 0
    output_notes = []
    for pattern in prediction_output:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Guitar()
        output_notes.append(new_note)
        offset += .5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp= 'lstm.midi')

run(midi_list)
