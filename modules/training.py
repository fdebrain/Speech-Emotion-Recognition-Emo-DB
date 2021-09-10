import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, make_scorer
from skopt import BayesSearchCV

# Allow for parallellization during training (Default: -1). Set a value of 1 to disable.
N_JOBS = -1


def score(y_true, y_pred):
    """Compute score based on true and predicted labels.

    Arguments:
        y_true {np.ndarray} -- True labels.
        y_pred {np.ndarray} -- Predicted labels (should be probabilities).

    Returns:
        score {float} -- Score metric.
    """
    return f1_score(y_true, y_pred, average='weighted')


def compute_score(model, X, y):
    """Compute score based on fitted model, input features and true labels.

    Arguments:
        model {sklearn estimator} -- Fitted model.
        X {np.ndarray} -- Input features.
        y {np.ndarray} -- True labels.

    Returns:
        score {float} -- Score metric.
    """
    y_pred = model.predict(X)
    return score(y, y_pred)


def train(X_train, y_train, speaker_ids, model, search_space, mode, n_iters=20, seed=None):
    """Fit and optimize model via Bayesian search group k-fold cross-validation on training data.

    Arguments:
        X_train {np.ndarray} -- Training features.
        y_train {np.ndarray} -- Training labels.
        speaker_ids {np.ndarray} -- Training speaker ids.
        model {sklearn estimator} -- Instanciated estimator to fit.
        search_space {dict} -- Search interval for each parameter of model to optimize.
        mode {str} --

    Keyword Arguments:
        n_iters {int} -- Number of iterations allowed during optimization (default: {20}).
        seed {int or None} -- Allow for reproducible results if int. Allow for randomness if None (default: {None}).

    Returns:
        fitted_model {sklearn estimator} -- Fitted estimator.
        cv_mean, cv_std (mode=='optimize') {float, float} -- Cross-validated F1 mean and standard variation over k-folds.
        y_preds (mode=='validate') -- Cross-validated predictions over k-folds.
        training_time {float} -- Time if took to fit the model.
    """
    start_time = time.time()
    kf = GroupKFold(n_splits=len(set(speaker_ids)))

    if mode == 'optimize':
        fitted_model = BayesSearchCV(model, search_space, cv=kf, refit=True,
                                     scoring=make_scorer(score),
                                     n_jobs=N_JOBS, n_iter=n_iters,
                                     random_state=seed, verbose=False)
        fitted_model.fit(X_train, y_train, groups=speaker_ids)

        # Extract cv validation scores
        cv_mean = fitted_model.cv_results_[
            'mean_test_score'][fitted_model.best_index_]
        cv_std = fitted_model.cv_results_[
            'std_test_score'][fitted_model.best_index_]
        training_time = time.time() - start_time

        return fitted_model, cv_mean, cv_std, training_time

    elif mode == 'validate':
        y_preds = cross_val_predict(model, X_train, y_train,
                                    cv=kf, groups=speaker_ids,
                                    n_jobs=N_JOBS)
        fitted_model = model.fit(X_train, y_train)

        training_time = time.time() - start_time
        return fitted_model, y_preds, training_time


def compute_cm(y_true, y_pred, classes, normalize=False, title=None):
    """Compute confusion matrix based on true and predicted labels."""
    fig, ax = plt.subplots(figsize=(4, 4))

    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(ticks=np.arange(cm.shape[1]))
    ax.set_yticks(ticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(labels=classes)
    ax.set_yticklabels(labels=classes)
    ax.tick_params(labelrotation=45)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
