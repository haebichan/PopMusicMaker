import numpy as np
from flask import Flask, abort, jsonify, request, render_template, send_file
import pickle
import make_song
import make_lyrics
import make_random_lyrics
import json

app = Flask(__name__)



@app.route('/', methods=['GET','POST'])
def song():
    if request.method == 'GET':
        return render_template('home.html')

    midi_list = request.form.getlist('songs')

    print('midi_list:', midi_list)

    make_song.main(midi_list)

    return send_file('your_song.midi')



@app.route('/lyrics', methods=['GET','POST'])
def lyrics():
    if request.method == 'GET':
        return render_template('home.html')

    word_list = request.form.getlist('user_lyrics')
    print('word_list: ', word_list)

    final_word_list = []
    for i in word_list:
        j = i.split(',')
        for word in j:
            word = word.strip()
            final_word_list.append(word)

    make_lyrics.create_user_lyrics(final_word_list)

    return render_template('home.html')

@app.route('/random_lyrics', methods=['GET','POST'])
def random_lyrics():
    if request.method == 'GET':
        return render_template('home.html')

    word_list = request.form.getlist('machine_random_input')

    make_random_lyrics.create_user_lyrics(word_list)

    return render_template('home.html')

# @app.route('/play_song', methods = ['GET'])
# def play_downloaded_song():
#     return send_file('your_song.midi')




if __name__ == '__main__':
    app.run(port=3000, debug=True)