import os
import logging

from flask import Flask, request, jsonify
from modules.formatter import label2emotion
from modules.models import set_seed
from modules.pipeline import (load_data, preprocess, prepare_data,
                              train_base_models, train_ensemble_model,
                              load_query, extract_features,
                              save_pipeline, load_pipeline)

# Fix random generator for reproducibility. Set to None to allow randomness.
SEED = 42
set_seed(SEED)

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

app = Flask(__name__)
pipeline_filename = 'final_model.pkl'


@app.route('/train', methods=['GET'])
def train():
    """API GET request to train and save ensemble model.
    Example: curl http://0.0.0.0:8080/train?n=3
    n is an optional parameter that define the number of top base models to consider.
    """
    query_parameters = request.args
    n_best = int(query_parameters.get('n', 3))

    df = load_data(filename='train_data.pkl', seed=SEED)
    speaker_ids = df['speaker']
    X_train, y_train = prepare_data(df)
    preproc, X_train = preprocess(X_train)
    base_models = train_base_models(X_train, y_train, speaker_ids, SEED)
    final_model = train_ensemble_model(base_models, X_train, y_train,
                                       speaker_ids, n_best, SEED)
    save_pipeline(preproc, final_model, pipeline_filename)
    return 'Success !\n'


@app.route('/test_ids', methods=['GET'])
def fetch_test_id():
    """API GET request to fetch .wav filenames for which the model wasn't trained on.
    Example: curl http://0.0.0.0:8080/test_ids
    """
    df = load_data(filename='test_data.pkl', seed=SEED)
    return jsonify(ids=list(df['filename']))


@app.route('/predict', methods=['GET'])
def predict():
    """API GET request to predict emotional state of a query .wav file using fitted ensemble model.
    Example: curl http://0.0.0.0:8080/predict?id=03a01Fa
    id is the .wav filename to query (without '.wav').
    """
    query_parameters = request.args
    query_filename = str(query_parameters.get('id'))

    query_signal = load_query(query_filename)

    if query_signal is None:
        return 'Cannot find query file ! \n'

    if not os.path.isfile(pipeline_filename):
        return 'Please train first ! \n'

    preproc, model = load_pipeline(pipeline_filename)
    X_query = extract_features(query_signal, mfcc=True)
    X_query = preproc.transform(X_query.reshape((1, -1)))

    y_pred = model.predict(X_query)
    return jsonify(prediction=label2emotion[int(y_pred)])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
