import librosa
import numpy as np


def compute_stft(signal, n_fft=2048):
    """Compute mean Short-Time Fourier Transform in dB from 1D signal.
    This feature extracts the signal's Fourier spectrum computed on short segments.

    Arguments:
        signal {np.ndarray} -- Real valued input signal. Shape (n,).

    Returns:
        X {np.ndarray} -- Extracted STFT features expressed in dB and averaged over time axis. Shape: (1 + n_fft/2,).
    """
    X = librosa.stft(signal, n_fft)
    Xdb = librosa.amplitude_to_db(abs(X), ref=np.max)
    return np.mean(Xdb, axis=1)


def compute_mfccs(signal, n_mfcc):
    """Compute mean Mel-frequency cepstral coefficients from 1D signal.
    This feature describes to the shape of the signal's spectral envelope.

    Arguments:
        signal {np.ndarray} -- Real valued input signal. Shape: (n,).
        n_mfcc {int} -- Number of MFCC to extract.

    Returns:
        X {np.ndarray} -- Extracted MFCC features averaged over time axis. Shape: (n_mfcc,).
    """
    mfcc = librosa.feature.mfcc(signal, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def compute_zcr(signal):
    """Compute Zero-Crossing Rate from 1D signal.
    This feature correspond to the local rate of sign-changes along a signal.

    Arguments:
        signal {np.ndarray} -- Real valued input signal. Shape: (n,).

    Returns:
        X -- Extracted ZCR feature. Shape: (m,) with m ~ n/512.
    """
    return librosa.feature.zero_crossing_rate(signal).flatten()


def compute_zc(signal):
    """Compute number of Zero-Crossings from 1D signal.

    Arguments:
        signal {np.ndarray} -- Real valued input signal. Shape: (n,).

    Returns:
        x {int} -- Extracted ZC feature.
    """
    return librosa.zero_crossings(signal).sum()


def compute_chroma(signal):
    """Compute chromagram from a 1D signal.
    This feature is very similar to STFT but its representation os based on musical octaves using a chromatic scale.

    Arguments:
        signal {np.ndarray} -- Real valued input signal. Shape: (n,).

    Returns:
        X -- Extracted chroma feature. Shape:
    """
    stft = librosa.stft(signal)
    stft = librosa.amplitude_to_db(abs(stft), ref=np.max)
    chroma = librosa.feature.chroma_stft(S=stft)
    return np.mean(chroma, axis=1)


def compute_mel(signal, n_mels=128):
    """Compute Mel-scale spectrogram from a 1D signal.
    This feature is very similar to STFT but its representation is closer to the human ear using Mel-scale.

    Arguments:
        signal {np.ndarray} -- Real valued input signal. Shape: (n,).

    Returns:
        X -- Extracted Mel feature. Shape: (n_mels,).
    """
    mel = librosa.feature.melspectrogram(signal, n_mels)
    return np.mean(mel, axis=1)


def extract_features(signal, zc=False, energy=False, stft=False, mfcc=False, chroma=False, mel=False):
    """ Extract and stack desired features extracted from a 1D signal.
    Credits: https://github.com/ritikraj660/Speech-emotion-recognition/blob/master/Speech%20Emotion%20Recognition.ipynb
    """
    X_feat = np.array([])
    if zc:
        feat = compute_zc(signal)
        X_feat = np.hstack((X_feat, feat))
    if stft:
        feat = compute_stft(signal)
        X_feat = np.hstack((X_feat, feat))
    if chroma:
        feat = compute_chroma(signal)
        X_feat = np.hstack((X_feat, feat))
    if mfcc:
        feat = compute_mfccs(signal, n_mfcc=20)
        X_feat = np.hstack((X_feat, feat))
    if mel:
        feat = compute_mel(signal)
        X_feat = np.hstack((X_feat, feat))
    return X_feat
