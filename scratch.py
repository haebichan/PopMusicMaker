import pandas as pd
import numpy as np
from music21 import converter, chord, note, instrument, stream
from collections import Counter
import random

pd.set_option('display.max_columns', 999)


def main(input):
    midi_list = input

    def get_midi(midi_list):
        all_midis = []
        all_parts = []

        for song in midi_list:
            midi = converter.parse('/var/www/FlaskApps/PopMusicMakerApp/midi/' + song)
            for i in midi.parts:
                i.insert(0, instrument.Piano())
            parts = instrument.partitionByInstrument(midi)

            all_midis.append(midi)
            all_parts.append(parts)

        return all_midis, all_parts

    all_midis, all_parts = get_midi(midi_list)

    def get_notes_offset_durations(all_parts):
        notes = []
        notes_offset = []
        durations = []

        for parts in all_parts:
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

        return notes, notes_offset, durations

    notes, notes_offset, durations = get_notes_offset_durations(all_parts)

    def get_harmony(midi_list):
        harmony = []
        harmony_duration = []

        for song in midi_list:
            midi = converter.parse('/var/www/FlaskApps/PopMusicMakerApp/midi/' + song)

            for i in midi[1].recurse():
                if isinstance(i, note.Note):
                    harmony.append(str(i.pitch))
                    harmony_duration.append(i.duration.quarterLength)
                elif isinstance(i, chord.Chord):
                    harmony_duration.append(i.duration.quarterLength)

                    i = str(i).replace('>', '')
                    chords = '|'.join(i.split()[1:])
                    harmony.append(chords)

        return harmony, harmony_duration

    harmony, harmony_duration = get_harmony(midi_list)
    allnotes = set(note for note in notes)
    keys = list(set(key for key in harmony))
    harmony_duration = [4.0 for i in harmony_duration]

    def create_vertical_dependency_dictionary(keys, notes, notes_offset):
        dic = {}

        for key in keys:
            dic[key] = []

        for i in range(len(notes)):
            if notes[i] not in keys:
                continue

            j = i + 1
            while j < len(notes) and np.abs(notes_offset[i] - notes_offset[j]) <= 4.0:
                dic[notes[i]].append(notes[j])
                j += 1

        return dic

    vertical_dependency_dictionary = create_vertical_dependency_dictionary(keys, notes, notes_offset)

    def create_dependency_map(dic, notes):
        keys_notes_map = {}

        for key in keys:
            keyMap = {}
            for note in set(notes):
                keyMap[note] = 0.0
            keys_notes_map[key] = keyMap

        for key in keys:
            my_notes = dic[key]
            note_count = Counter(my_notes)
            for my_note_count in note_count.keys():
                keys_notes_map[key][my_note_count] = note_count[my_note_count] / len(my_notes)

        return keys_notes_map

    vert_keys_notes_map = create_dependency_map(vertical_dependency_dictionary, notes)

    vert_dep_matrix = pd.DataFrame(vert_keys_notes_map).T

    def get_melody_offset_durations(instrument_parts):
        melody = []
        melody_offset = []
        melody_durations = []

        for parts in instrument_parts:
            for i in parts.recurse():
                if isinstance(i, note.Note):
                    melody.append(str(i.pitch))
                    melody_offset.append(float(i.offset))
                    melody_durations.append(float(i.duration.quarterLength))

                elif isinstance(i, chord.Chord):
                    melody_offset.append(float(i.offset))
                    melody_durations.append(float(i.duration.quarterLength))

                    i = str(i).replace('>', '')
                    chords = '|'.join(i.split()[1:])
                    melody.append(chords)

        return melody, melody, melody_durations

    melody, melody_offset, melody_durations = get_melody_offset_durations(all_parts)

    all_melody_notes = set(note for note in melody)

    def create_horizontal_dependency_dictionary(all_melody_notes, melody, melody_durations):
        hori_dic = {}
        count = 0

        for melody_note in all_melody_notes:
            hori_dic[melody_note] = []

        for note, next_note, duration in zip(list(melody), list(melody)[1:], melody_durations):
            if count <= 4:
                hori_dic[note].append(next_note)
                count += duration
            count = 0

        return hori_dic

    hori_dic = create_horizontal_dependency_dictionary(all_melody_notes, melody, melody_durations)

    def create_horizontal_dependency_map(hori_dic):
        horizontal_dependency_map = {}

        for y_axis_key in hori_dic.keys():
            hori_map = {}
            for x_axis_key in hori_dic.keys():
                hori_map[x_axis_key] = 0.0
            horizontal_dependency_map[y_axis_key] = hori_map

        for y_axis_key in hori_dic.keys():
            new_notes = hori_dic[y_axis_key]
            new_note_count = Counter(new_notes)
            for my_note_count in new_note_count.keys():
                horizontal_dependency_map[y_axis_key][my_note_count] = new_note_count[my_note_count] / len(new_notes)

        return horizontal_dependency_map

    horizontal_dependency_map = create_horizontal_dependency_map(hori_dic)
    hori_dep_matrix = pd.DataFrame(horizontal_dependency_map).T

    hori_dep_matrix = hori_dep_matrix.drop(hori_dep_matrix[hori_dep_matrix.sum(1).values == 0.0].index)

    all_harmony_notes = set(note for note in harmony)
    harmony_hori_dic = create_horizontal_dependency_dictionary(all_harmony_notes, harmony, harmony_duration)

    harmony_horizontal_dependency_map = create_horizontal_dependency_map(harmony_hori_dic)

    harmony_hori_dep_matrix = pd.DataFrame(harmony_horizontal_dependency_map).T

    harmony_hori_dep_matrix = harmony_hori_dep_matrix.drop(
        harmony_hori_dep_matrix[harmony_hori_dep_matrix.sum(1).values == 0.0].index)

    def create_vert_duration_dictionary(matrix, notes):
        vert_duration_dictionary = {}

        for single_note in list(matrix.columns):
            vert_duration_dictionary[single_note] = []

        for note, duration in zip(notes, durations):
            vert_duration_dictionary[note].append(duration)

        return vert_duration_dictionary

    def create_hori_duration_dictionary(matrix, melody, melody_durations):
        hori_duration_dictionary = {}

        for single_note in list(matrix.columns):
            hori_duration_dictionary[single_note] = []

        for note, duration in zip(melody, melody_durations):
            hori_duration_dictionary[note].append(duration)

        return hori_duration_dictionary

    vert_duration_dictionary = create_vert_duration_dictionary(vert_dep_matrix, notes)

    hori_duration_dictionary = create_hori_duration_dictionary(hori_dep_matrix, melody, melody_durations)

    harmony_hori_duration_dictionary = create_hori_duration_dictionary(harmony_hori_dep_matrix, harmony,
                                                                       harmony_duration)

    def create_vert_duration_map(allnotes, durations, duration_dictionary):
        duration_notes_map = {}

        for single_note in allnotes:
            durationMap = {}
            for each_duration in durations:
                durationMap[each_duration] = 0.0
            duration_notes_map[single_note] = durationMap

        for single_note in allnotes:
            note_duration = duration_dictionary[single_note]
            note_duration_count = Counter(note_duration)
            for single_note_duration in note_duration_count.keys():
                duration_notes_map[single_note][single_note_duration] = note_duration_count[single_note_duration] / len(
                    note_duration)

        return duration_notes_map

    vert_duration_notes_map = create_vert_duration_map(allnotes, durations, vert_duration_dictionary)

    vert_duration_matrix = pd.DataFrame(vert_duration_notes_map).T

    def create_hori_duration_map(all_melody_notes, melody_durations, hori_duration_dictionary):
        duration_notes_map = {}

        for single_note in all_melody_notes:
            durationMap = {}
            for each_duration in melody_durations:
                durationMap[each_duration] = 0.0
            duration_notes_map[single_note] = durationMap

        for single_note in all_melody_notes:
            note_duration = hori_duration_dictionary[single_note]
            note_duration_count = Counter(note_duration)
            for single_note_duration in note_duration_count.keys():
                duration_notes_map[single_note][single_note_duration] = note_duration_count[single_note_duration] / len(
                    note_duration)

        return duration_notes_map

    hori_duration_notes_map = create_hori_duration_map(all_melody_notes, melody_durations, hori_duration_dictionary)

    hori_duration_matrix = pd.DataFrame(hori_duration_notes_map).T

    harmony_hori_duration_map = create_hori_duration_map(all_harmony_notes, harmony_duration,
                                                         harmony_hori_duration_dictionary)

    harmony_hori_duration_matrix = pd.DataFrame(harmony_hori_duration_map).T

    harmony_hori_duration_matrix = harmony_hori_duration_matrix.rename(
        columns={'1/3': '0.33', '2/3': '0.66', '4/3': '1.33', '5/3': '1.66', '7/3': '2.33', '8/3': '2.66',
                 '10/3': '3.33', '11/3': '3.66', '16/3': '5.33', '22/3': '7.33', '23/3': '7.66', '25/3': '8.33'})

    every_note = ['C1', 'D-1', 'D1', 'E-1', 'E1', 'F-1', 'F1', 'G1', 'A-1', 'A1', 'B-1', 'B1', 'C-2', 'C2', 'D-2', 'D2',
                  'E-2', 'E2', 'F-2', 'F2', 'G-2', 'G2', 'A-2', 'A2', 'B-2', 'B2', 'C-3', 'C3', 'D-3', 'D3', 'E-3',
                  'E3', 'F-3', 'F3', 'G-3', 'G3', 'A-3', 'A3', 'B-3', 'B3', 'C-4', 'C4', 'D-4', 'D4', 'E-4', 'E4',
                  'F-4', 'F4', 'G-5', 'G4', 'A-4', 'A4', 'B-4', 'B4', 'C-5', 'C5', 'D-5', 'D5', 'E-5', 'E5', 'F-5',
                  'F5', 'G-5', 'G5', 'A-5', 'A5', 'B-5', 'B5', 'C-6', 'C6', 'D-6', 'D6', 'E-6', 'E6', 'F-6', 'F6',
                  'G-6', 'G6', 'A-6', 'A6', 'B-6', 'B6', 'C-7', 'C7', 'D-7', 'D7', 'E-7', 'E7', 'F-7', 'F7', 'G-7',
                  'G7']
    every_note_number = [i for i in range(len(every_note))]

    every_note_dic = {}

    for i, j in zip(every_note_number, every_note):
        every_note_dic[j] = i

    for i in hori_dep_matrix.index:
        if len(i) > 3:
            split_note = i.split('|')
            every_note_dic[i] = every_note_dic[split_note[0]]

    def get_harmony(midi_list):
        harmony = []
        harmony_duration = []

        for song in midi_list:
            midi = converter.parse('/var/www/FlaskApps/PopMusicMakerApp/midi/' + song)

            for i in midi[1].recurse():
                if isinstance(i, note.Note):
                    harmony.append(str(i.pitch))
                    harmony_duration.append(i.duration)
                elif isinstance(i, chord.Chord):
                    harmony.append('|'.join(i.pitchNames))
                    harmony_duration.append(i.duration)

        return harmony, harmony_duration

    harmony, harmony_duration = get_harmony(midi_list)

    harmony_duration = [4.0 for i in harmony_duration]

    def create_harmony_list(harmony_hori_dep_matrix, harmony_duration, harmony_offset_count_number=64):
        harmony_list = []
        harmony_duration_list = []

        harmony_offset_count = 0

        harmony_note = random.choice(list(harmony_hori_dep_matrix.index))
        harmony_duration = float(np.random.choice(harmony_hori_duration_matrix.loc[harmony_note].index,
                                                  p=harmony_hori_duration_matrix.loc[harmony_note].values))

        # If harmony duration is 2, then 64 means that the harmony will repeat after 8 harmony notes
        while harmony_offset_count <= harmony_offset_count_number and (
                harmony_duration + harmony_offset_count) < harmony_offset_count_number:
            harmony_list.append(harmony_note)
            harmony_duration_list.append(harmony_duration)

            harmony_offset_count += harmony_duration

            harmony_note = np.random.choice(harmony_hori_dep_matrix.loc[harmony_note].index,
                                            p=harmony_hori_dep_matrix.loc[harmony_note].values)
            harmony_duration = float(np.random.choice(harmony_hori_duration_matrix.loc[harmony_note].index,
                                                      p=harmony_hori_duration_matrix.loc[harmony_note].values))

        return harmony_list, harmony_duration_list

    harmony_list, harmony_duration_list = create_harmony_list(harmony_hori_dep_matrix, harmony_duration)

    harmony_offset_count_number = 64

    harmony_duration_list[-1] = (harmony_offset_count_number - sum(harmony_duration_list[:-1]))

    harmony_duration_list[-1] = harmony_duration[0]

    highest_harmony_number = []
    for harmony_note in harmony_list:

        if len(harmony_note) > 2:

            split_harmony = harmony_note.split('|')
            highest_harmony_number.append(split_harmony[1])


        else:
            highest_harmony_number.append(harmony_note)

    highest_harmony_number = [every_note_dic[i] for i in highest_harmony_number]

    highest_harmony_number = max(highest_harmony_number)

    def create_song(vert_dep_matrix, vert_duration_matrix, hori_dep_matrix, hori_duration_matrix, harmony_list, harmony_duration_list, gap_width=10, gap_width2=15, gap_width3=20):

       song_generation = []
        song_generation_duration = []

        song_generation2 = []
        song_generation_duration2 = []

        song_generation3 = []
        song_generation_duration3 = []

        segment_list = [2, 1, 3]

        for segment_count in segment_list:

            if segment_count == 2:

                repetition_count = 0

                while repetition_count <= segment_count:

                    for harmony_note, harmony_duration in zip(harmony_list[::2], harmony_duration_list[::2]):
                        song_generation.append(harmony_note)
                        song_generation_duration.append(harmony_duration)

                        melody_note = np.random.choice(vert_dep_matrix.loc[harmony_note].index,
                                                       p=vert_dep_matrix.loc[harmony_note].values)
                        melody_duration = float(np.random.choice(vert_duration_matrix.loc[harmony_note].index,
                                                                 p=vert_duration_matrix.loc[harmony_note].values))

                        while every_note_dic[melody_note] < (highest_harmony_number + gap_width):
                            melody_note = np.random.choice(vert_dep_matrix.loc[harmony_note].index,
                                                           p=vert_dep_matrix.loc[harmony_note].values)
                            melody_duration = float(np.random.choice(vert_duration_matrix.loc[harmony_note].index,
                                                                     p=vert_duration_matrix.loc[harmony_note].values))

                        melody_dic = {}
                        count = 0

                        while count <= 8.0 and (melody_duration + count <= 8.0):

                            melody_dic[melody_note] = melody_duration

                            count += melody_duration

                            melody_note = np.random.choice(hori_dep_matrix.loc[melody_note].index,
                                                           p=hori_dep_matrix.loc[melody_note].values)
                            melody_duration = float(np.random.choice(hori_duration_matrix.loc[melody_note].index,
                                                                     p=hori_duration_matrix.loc[melody_note].values))

                            while every_note_dic[melody_note] < (highest_harmony_number + gap_width):
                                melody_note = np.random.choice(hori_dep_matrix.loc[melody_note].index,
                                                               p=hori_dep_matrix.loc[melody_note].values)
                                melody_duration = float(np.random.choice(hori_duration_matrix.loc[melody_note].index,
                                                                         p=hori_duration_matrix.loc[
                                                                             melody_note].values))

                            melody_dic[melody_note] = melody_duration
                            count += melody_duration

                        song_generation.append(melody_dic)

                    repetition_count += 1
                segment_count += 1

            elif segment_count == 1:

                repetition_count = 0

                while repetition_count <= segment_count:

                    for harmony_note, harmony_duration in zip(harmony_list[::2], harmony_duration_list[::2]):
                        song_generation2.append(harmony_note)
                        song_generation_duration2.append(harmony_duration)

                        melody_note = np.random.choice(vert_dep_matrix.loc[harmony_note].index,
                                                       p=vert_dep_matrix.loc[harmony_note].values)
                        melody_duration = float(np.random.choice(vert_duration_matrix.loc[harmony_note].index,
                                                                 p=vert_duration_matrix.loc[harmony_note].values))

                        while every_note_dic[melody_note] < (highest_harmony_number + gap_width2):
                            melody_note = np.random.choice(vert_dep_matrix.loc[harmony_note].index,
                                                           p=vert_dep_matrix.loc[harmony_note].values)
                            melody_duration = float(np.random.choice(vert_duration_matrix.loc[harmony_note].index,
                                                                     p=vert_duration_matrix.loc[harmony_note].values))

                        melody_dic = {}
                        count = 0

                        while count <= 8.0 and (melody_duration + count <= 8.0):

                            melody_dic[melody_note] = melody_duration

                            count += melody_duration

                            melody_note = np.random.choice(hori_dep_matrix.loc[melody_note].index,
                                                           p=hori_dep_matrix.loc[melody_note].values)
                            melody_duration = float(np.random.choice(hori_duration_matrix.loc[melody_note].index,
                                                                     p=hori_duration_matrix.loc[melody_note].values))

                            while every_note_dic[melody_note] < (highest_harmony_number + gap_width2):
                                melody_note = np.random.choice(hori_dep_matrix.loc[melody_note].index,
                                                               p=hori_dep_matrix.loc[melody_note].values)
                                melody_duration = float(np.random.choice(hori_duration_matrix.loc[melody_note].index,
                                                                         p=hori_duration_matrix.loc[
                                                                             melody_note].values))

                            melody_dic[melody_note] = melody_duration
                            count += melody_duration

                        song_generation2.append(melody_dic)

                    repetition_count += 1
                segment_count += 1



            elif segment_count == 3:

                repetition_count = 0

                while repetition_count <= segment_count:

                    for harmony_note, harmony_duration in zip(harmony_list[::2], harmony_duration_list[::2]):
                        song_generation3.append(harmony_note)
                        song_generation_duration3.append(harmony_duration)

                        melody_note = np.random.choice(vert_dep_matrix.loc[harmony_note].index,
                                                       p=vert_dep_matrix.loc[harmony_note].values)
                        melody_duration = float(np.random.choice(vert_duration_matrix.loc[harmony_note].index,
                                                                 p=vert_duration_matrix.loc[harmony_note].values))

                        while every_note_dic[melody_note] < (highest_harmony_number + gap_width3):
                            melody_note = np.random.choice(vert_dep_matrix.loc[harmony_note].index,
                                                           p=vert_dep_matrix.loc[harmony_note].values)
                            melody_duration = float(np.random.choice(vert_duration_matrix.loc[harmony_note].index,
                                                                     p=vert_duration_matrix.loc[harmony_note].values))

                        melody_dic = {}
                        count = 0

                        while count <= 8.0 and (melody_duration + count <= 8.0):

                            melody_dic[melody_note] = melody_duration

                            count += melody_duration

                            melody_note = np.random.choice(hori_dep_matrix.loc[melody_note].index,
                                                           p=hori_dep_matrix.loc[melody_note].values)
                            melody_duration = float(np.random.choice(hori_duration_matrix.loc[melody_note].index,
                                                                     p=hori_duration_matrix.loc[melody_note].values))

                            while every_note_dic[melody_note] < (highest_harmony_number + gap_width3):
                                melody_note = np.random.choice(hori_dep_matrix.loc[melody_note].index,
                                                               p=hori_dep_matrix.loc[melody_note].values)
                                melody_duration = float(np.random.choice(hori_duration_matrix.loc[melody_note].index,
                                                                         p=hori_duration_matrix.loc[
                                                                             melody_note].values))

                            melody_dic[melody_note] = melody_duration
                            count += melody_duration

                        song_generation3.append(melody_dic)

                    repetition_count += 1
                segment_count += 1

        return song_generation, song_generation_duration, song_generation2, song_generation_duration2, song_generation3, song_generation_duration3

    song, song_duration, song2, song_duration2, song3, song_duration3 = create_song(vert_dep_matrix,
                                                                                    vert_duration_matrix,
                                                                                    hori_dep_matrix,
                                                                                    hori_duration_matrix, harmony_list,
                                                                                    harmony_duration_list)

    song = song + song2 + song3 + song2 + song3
    song_duration = song_duration + song_duration2 + song_duration3 + song_duration2 + song_duration3

    score = stream.Score()
    p1 = stream.Part()
    p1.id = 'harmony'

    p2 = stream.Part()
    p2.id = 'melody'

    for harmony_note, h_duration, melody_note in zip(song[::2], song_duration, song[1::2]):

        if len(harmony_note) > 2:
            harmony_chord = chord.Chord(harmony_note.split('|'))
            harmony_chord.quarterLength = h_duration
            p1.append(harmony_chord)
        else:
            n = note.Note(harmony_note)
            n.quarterLength = h_duration
            p1.append(n)

        if bool(melody_note) == False:
            rest = note.Rest()
            rest.quarterLength = h_duration
            p2.append(rest)

        else:
            for individual_note, individual_note_duration in melody_note.items():

                if "|" not in individual_note:
                    ind_note = note.Note(individual_note)
                    ind_note.quarterLength = individual_note_duration
                    p2.append(ind_note)
                else:
                    split_notes = individual_note.split("|")
                    chords = chord.Chord(split_notes)
                    chords.quarterLength = individual_note_duration

                    p2.append(chords)

    score.insert(0, p2)
    score.insert(0, p1)

    for i in score.parts:
        i.insert(0, instrument.Piano())

    score.write('midi', fp='your_song.midi')
