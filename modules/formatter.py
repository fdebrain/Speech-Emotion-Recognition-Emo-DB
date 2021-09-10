import logging

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

# Label encoding (filename/human-readable/encoded-label)
letter2emotion = {'W': 'Anger', 'L': 'Boredom', 'E': 'Disgust',
                  'A': 'Fear', 'F': 'Hapiness', 'T': 'Sadness',
                  'N': 'Neutral'}
emotion2label = dict(zip(letter2emotion.values(),
                         range(len(letter2emotion.values()))))
label2emotion = dict(zip(range(len(letter2emotion.values())),
                         letter2emotion.values()))


def extract_label(filename):
    """Extract emotion state and encoded value of label from .wav filename.

    Arguments:
        filename {Path} -- Name of .wav file from which to extract label.

    Returns:
        emotion {str} -- Emotional state of speaker in .wav file.
        label {int} -- Encoded label.
        speaker {int} -- Speaker id.
    """
    letter = filename.parts[-1][5]
    emotion = letter2emotion[letter]
    label = emotion2label[emotion]
    speaker = filename.parts[-1][:2]
    return emotion, label, speaker


def format_dataset(datafolder=Path('./data/wav'), seed=None):
    """Load .wav files, extract labels and split into train and test dataframes.

    Keyword Arguments:
        datafolder {Path} -- Path of .wav files directory (default: {Path('./data/wav')}).
        seed {int or None} -- Fix int seed for reproducibility. Allow for randomness with None value (default: {None}).
    """
    wavfiles = list(datafolder.glob('*.wav'))
    logging.info(f'Found {len(wavfiles)} .wav files !')

    logging.info('Formatting data...')
    data = []
    for file in wavfiles:
        x, sr = librosa.load(file)
        emotion, label, speaker = extract_label(file)
        data.append({'filename': str(file), 'signal': np.array(x),
                     'len': len(x), 'emotion': str(emotion),
                     'speaker': int(speaker), 'label': int(label)})
    df = pd.DataFrame(data)

    logging.info('Splitting training and testing data...')
    np.random.seed(seed)
    test_speaker_ids = np.random.choice(np.unique(df['speaker']),
                                        size=2, replace=False)
    df_train = df[~np.isin(df['speaker'], test_speaker_ids)]
    df_test = df[np.isin(df['speaker'], test_speaker_ids)]

    df_train.to_pickle('train_data.pkl')
    df_test.to_pickle('test_data.pkl')
    logging.info('Successfully saved formatted data !')
