import pandas as pd
import numpy as np
from collections import Counter
import random

def create_user_lyrics(user_input_list):

    zedd_df = pd.read_csv('/Users/Haebichan/Desktop/flask/Lyrics/zedd_lyrics.csv', index_col = 0, names = ['lyrics'], skiprows = [0])
    adele_df = pd.read_csv('/Users/Haebichan/Desktop/flask/Lyrics/adele_lyrics.csv', index_col = 0, names = ['lyrics'], skiprows = [0])
    avicii_df = pd.read_csv('/Users/Haebichan/Desktop/flask/Lyrics/avicii_lyrics.csv', index_col = 0, names = ['lyrics'], skiprows = [0])


    df = pd.concat([zedd_df, adele_df, avicii_df], 0)

    lyrics = [i for i in df.lyrics]

    lyrics = ''.join(lyrics)


    mystring = lyrics.replace('\'','')
    mystring = mystring.replace("",'')
    mystring = mystring.replace(",",'')
    mystring = mystring.replace("(",'')
    mystring = mystring.replace(")",'')
    mystring = mystring.replace("!",'')
    mystring = mystring.replace('''"''','')
    mystring = mystring.replace("?",'')
    mystring = mystring.replace("*",'')

    mylist = mystring.split('.')
    mylist = [i.strip() for i in mylist]
    mylist = [x for x in mylist if x]

    first_words = set()

    for i in mylist:
        j = i.split()
        first_words.add(j[0])



    all_words = []

    for i in mylist:
        j = i.split()
        for word in j:
            all_words.append(word)

    dic = {}

    for first_word in first_words:
        dic[first_word] = set()

    for i in mylist:
        j = i.split()
        if len(j) > 1:
            dic[j[0]].add(j[1])

    keys_notes_map = {}

    for key in first_words:
        keyMap = {}
        for note in set(all_words):
            keyMap[note] = 0.0
        keys_notes_map[key] = keyMap

    for key in first_words:
        my_notes = dic[key]
        note_count = Counter(my_notes)
        for my_note_count in note_count.keys():
            keys_notes_map[key][my_note_count] = note_count[my_note_count] / len(my_notes)


    vert_dep_matrix = pd.DataFrame(keys_notes_map).T

    hori_dic = {}

    for all_word in all_words:
        hori_dic[all_word] = set()

    for word, next_word in zip(all_words, all_words[1:]):
        hori_dic[word].add(next_word)


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


    hori_dic2 = {}

    for all_word in first_words:
        hori_dic2[all_word] = set()

    for word, next_word in zip(first_words, list(first_words)[1:]):
        hori_dic2[word].add(next_word)

    horizontal_dependency_map2 = create_horizontal_dependency_map(hori_dic2)

    hori_dep_matrix_first_word = pd.DataFrame(horizontal_dependency_map2).T

    hori_dep_matrix_first_word = hori_dep_matrix_first_word.iloc[:-1,:]

    hori_dep_matrix_first_word = hori_dep_matrix_first_word.drop(
        vert_dep_matrix[vert_dep_matrix.sum(1).values == 0.0].index)

    vert_dep_matrix = vert_dep_matrix.drop(vert_dep_matrix[vert_dep_matrix.sum(1).values == 0.0].index)


    #
    # first_word_list = []
    #
    # first_word = random.choice(list(hori_dep_matrix_first_word.index))
    #
    # for i in range(20):
    #     first_word_list.append(first_word)
    #
    #     first_word = np.random.choice(hori_dep_matrix_first_word.loc[first_word].index,
    #                                   p=hori_dep_matrix_first_word.loc[first_word].values)
    #

    first_word_list = user_input_list

    lyrics_generation = []

    for first_word in first_word_list:
        lyrics_generation.append(first_word)

        next_word = np.random.choice(vert_dep_matrix.loc[first_word].index, p=vert_dep_matrix.loc[first_word].values)

        lyrics_generation.append(next_word)

        for i in range(np.random.randint(4, 10)):
            next_word = np.random.choice(hori_dep_matrix.loc[next_word].index, p=hori_dep_matrix.loc[next_word].values)

            lyrics_generation.append(next_word)

    lyrics = []

    for i in lyrics_generation:
        if i in first_word_list:
            lyrics.append('. ')
        lyrics.append(i)

    lyrics = lyrics[1:]

    final_lyrics = u' '.join(lyrics).encode('utf-8').strip()

    final_lyrics = str(final_lyrics).split('.')

    final_lyrics = [i.strip() for i in final_lyrics]

    file = open('/Users/Haebichan/Desktop/test.txt', 'w')

    for each_line in final_lyrics:
        file.write("%s\n" % each_line)

