import os
from pathlib import Path
import joblib
import logging

import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler

from modules.formatter import format_dataset
from modules.feature_extraction import extract_features
from modules.models import (get_KNNC, get_SVC, get_RFC, get_MPC, get_XGBC,
                            get_VC, compute_weights)
from modules.training import train, score


def load_data(filename='train_data.pkl', seed=None):
    """Load dataset (format if necessary).

    Keyword Arguments:
        filename {str} -- Name of the dataset file (default: {'train_data.pkl'}).
        seed {int or None} -- Allow for reproducible data splitting when formatting (default: {None}).

    Returns:
        df -- Dataframe containing dataset to load.
    """
    if not os.path.isfile(filename):
        logging.info('Formatting data...')
        format_dataset(datafolder=Path('./data/wav'), seed=seed)

    logging.info('Loading data...')
    df = pd.read_pickle(filename)
    return df


def load_query(id):
    """Load .wav file to query.

    Arguments:
        id {str} -- Name of the query to load.

    Returns:
        signal {np.ndarray} -- Query signal.
    """
    file_path = f'data/wav/{id}.wav'
    logging.info(f'Loading query {file_path}...')

    if not os.path.isfile(file_path):
        logging.info(f'{id}.wav not found !')
        return None

    signal, _ = librosa.load(file_path)
    return signal


def preprocess(X_train):
    """Preprocess training features by scaling them.

    Arguments:
        X_train {np.ndarray} -- Stacked training features.

    Returns:
        scaler {sklearn transformer} -- Fitted MinMaxScaler.
        X_train {np.ndarray} -- Scaled training features.
    """
    logging.info('Scaling features...')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    return scaler, X_train


def prepare_data(df):
    """Extract features from dataframe signals.

    Arguments:
        df {pd.Dataframe} -- Dataframe containing dataset. Should contain a 'signal' column.

    Returns:
        X {np.ndarray} -- Extracted features.
        y {np.ndarray} -- Extracted encoded labels.
    """
    logging.info('Extracting features...')
    X = np.array([extract_features(x, mfcc=True)
                  for x in df['signal']])
    y = np.array(df['label'])
    return X, y


def train_base_models(X_train, y_train, speaker_ids, seed=None):
    """Fit base models on training data.

    Arguments:
        X_train {np.ndarray} -- Training features.
        y_train {np.ndarray} -- Training labels.
        speaker_ids {np.ndarray} -- Training speaker ids.

    Keyword Arguments:
        seed {int or None} -- Allow for reproducible results (default: {None}).

    Returns:
        trained_models {List[sklearn estimators]} -- Fitted base models.
    """
    logging.info('Training base models...')
    models = {'SVC': get_SVC(), 'RFC': get_RFC(), 'MPC': get_MPC(),
              'KNNC': get_KNNC(), 'XGBC': get_XGBC()}
    trained_models = []

    # Optimize base models via Stratified K-fold CV and Bayesian optimization (~2mn)
    for model_name, (model, search_space, n_iters) in models.items():
        pipeline, cv_mean, cv_std, training_time = train(X_train, y_train,
                                                         speaker_ids, model,
                                                         search_space,
                                                         mode='optimize',
                                                         n_iters=n_iters,
                                                         seed=seed)
        trained_models.append((model_name, pipeline.best_estimator_, cv_mean))
        logging.info(f'{model_name} fitted in {training_time:.1f}s')
        logging.info(f'Validation F1: {cv_mean:.2f}+-{cv_std:.2f} \n')
    return trained_models


def train_ensemble_model(base_models, X_train, y_train, speaker_ids, n_best=3, seed=None):
    """Fit ensemble model using n_best base models and training data.

    Arguments:
        base_models {List[sklearn estimators]} -- Fitted base models.
        X_train {np.ndarray} -- Training features.
        y_train {np.ndarray} -- Training labels.
        speaker_ids {np.ndarray} -- Training speaker ids.

    Keyword Arguments:
        n_best {int} -- Number of top base models to consider in ensemble model (default: {3}).
        seed {int or None} -- Allow for reproducible data splitting when formatting (default: {None}).

    Returns:
        trained_ensemble_model {sklearn estimator} -- Fitted ensemble model.
    """
    logging.info(f'Training ensemble model using {n_best} best base models...')

    # Sort models by mean CV score
    base_models.sort(key=lambda l: l[-1], reverse=True)

    # Extract n best base models (and drop scores)
    weights = compute_weights(base_models)[:n_best]
    base_models = [m[:2] for m in base_models[:n_best]]

    # Train ensemble model
    ensemble_model, _ = get_VC(base_models, weights)
    trained_ensemble_model, y_preds, training_time = train(X_train, y_train,
                                                           speaker_ids,
                                                           ensemble_model,
                                                           search_space=None,
                                                           mode='validate',
                                                           seed=seed)
    logging.info(f'Ensemble fitted in {training_time:.1f}s')
    logging.info(f'Validation F1: {score(y_train, y_preds):.2f}')
    return trained_ensemble_model


def save_pipeline(preproc, model, filename):
    """Save preprocessing step and fitted ensemble model.

    Arguments:
        preproc {sklearn transformer} -- Fitted preproc step.
        model {sklearn estimator} -- Fitted model.
        filename {str} -- Name of .pkl file to save.
    """
    logging.info('Saving ensemble model...')
    joblib.dump({'preproc': preproc, 'model': model},
                filename=filename)


def load_pipeline(filename):
    """Load preprocessing step and fitted model.

    Arguments:
        filename {str} -- Name of .pkl file to load.

    Returns:
        preproc {sklearn transformer} -- Fitted preproc step.
        model {sklearn estimator} -- Fitted model.
    """
    logging.info('Loading model...')
    pipeline = joblib.load(filename)
    return pipeline['preproc'], pipeline['model']
