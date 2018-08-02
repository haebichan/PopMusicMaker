#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wave
import numpy
import csv
import pickle
import librosa
import yaml
import soundfile

def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array

    Supports 24-bit wav-format, and flac audio through librosa.

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        # Load audio
        audio_data, sample_rate = soundfile.read(filename)
        audio_data = audio_data.T

        if mono:
            # Down-mix audio
            audio_data = numpy.mean(audio_data, axis=0)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate

    elif file_extension == '.flac':
        audio_data, sample_rate = librosa.load(filename, sr=fs, mono=mono)

        return audio_data, sample_rate

    return None, None


def load_event_list(file):
    """Load event list from tab delimited text file (csv-formated)

    Supported input formats:

        - [event_onset (float)][tab][event_offset (float)]
        - [event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
        - [file(string)[tab][scene_label][tab][event_onset (float)][tab][event_offset (float)][tab][event_label (string)]

    Event dict format:

        {
            'file': 'filename',
            'scene_label': 'office',
            'event_onset': 0.0,
            'event_offset': 1.0,
            'event_label': 'people_walking',
        }

    Parameters
    ----------
    file : str
        Path to the event list in text format (csv)

    Returns
    -------
    data : list of event dicts
        List containing event dicts

    """
    data = []
    with open(file, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) == 2:
                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1])
                    }
                )
            elif len(row) == 3:
                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1]),
                        'event_label': row[2]
                    }
                )
            elif len(row) == 4:
                data.append(
                    {
                        'file': row[0],
                        'event_onset': float(row[1]),
                        'event_offset': float(row[2]),
                        'event_label': row[3]
                    }
                )
            elif len(row) == 5:
                data.append(
                    {
                        'file': row[0],
                        'scene_label': row[1],
                        'event_onset': float(row[2]),
                        'event_offset': float(row[3]),
                        'event_label': row[4]
                    }
                )
    return data


def save_data(filename, data):
    """Save variable into a pickle file

    Parameters
    ----------
    filename: str
        Path to file

    data: list or dict
        Data to be saved.

    Returns
    -------
    nothing

    """

    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"))


def save_parameters(filename, parameters):
    """Save parameters to YAML-file

    Parameters
    ----------
    filename: str
        Path to file
    parameters: dict
        Dict containing parameters to be saved

    Returns
    -------
    Nothing

    """

    with open(filename, 'w') as outfile:
        outfile.write(yaml.dump(parameters, default_flow_style=False))


def load_parameters(filename):
    """Load parameters from YAML-file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    parameters: dict
        Dict containing loaded parameters

    Raises
    -------
    IOError
        file is not found.

    """

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return yaml.load(f)
    else:
        raise IOError("Parameter file not found [%s]" % filename)


def save_text(filename, text):
    """Save text into text file.

    Parameters
    ----------
    filename: str
        Path to file

    text: str
        String to be saved.

    Returns
    -------
    nothing

    """

    with open(filename, "w") as text_file:
        text_file.write(text)


def load_text(filename):
    """Load text file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    text: string
        Loaded text.

    """

    with open(filename, 'r') as f:
        return f.readlines()
