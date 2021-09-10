import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Fix random generator for reproducibility
SEED = None


def set_seed(seed):
    """Fix the seed for reproducible results."""
    global SEED
    SEED = seed


def get_SVC():
    """ Instanciate Support Vector Classifier.
    Credits: https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769

    Returns:
        model {sklearn estimator} -- Instanciated model.
        search_space {dict} -- Search interval for each parameter of model to optimize.
        n_iters {int} -- Number of iterations allowed during optimization.
    """
    search_space = {'C': (1e-3, 1e3, 'log-uniform'),
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': [0.1, 1, 10, 100, 'auto'],
                    'degree': [0, 1, 2, 3, 4, 5, 6]}
    model = SVC(random_state=SEED)
    n_iters = 20
    return model, search_space, n_iters


def get_RFC():
    """ Instanciate Random Forest Classifier.
    Credits: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    Returns:
        model {sklearn estimator} -- Instanciated model.
        search_space {dict} -- Search interval for each parameter of model to optimize.
        n_iters {int} -- Number of iterations allowed during optimization.
    """
    search_space = {'n_estimators': [50, 100, 250, 500],
                    'bootstrap': [True, False],
                    'max_depth': [10, 25, 50, None],
                    'min_samples_leaf': (1, 5),
                    'min_samples_split': (2, 10)}
    model = RandomForestClassifier(random_state=SEED)
    n_iters = 20
    return model, search_space, n_iters


def get_XGBC():
    """ Instanciate XGBoost Classifier.
    Credits: https://www.datasciencelearner.com/gradient-boosting-hyperparameters-tuning/

    Returns:
        model {sklearn estimator} -- Instanciated model.
        search_space {dict} -- Search interval for each parameter of model to optimize.
        n_iters {int} -- Number of iterations allowed during optimization.
    """
    search_space = {'eta': [0.3],
                    'max_depth': (3, 10),
                    'subsample': (0.5, 1.0),
                    'colsample_bytree': (0.5, 1.0)}
    model = XGBClassifier(objective='multi:softmax', random_state=SEED)
    n_iters = 15
    return model, search_space, n_iters


def get_KNNC():
    """ Instanciate K-Nearest Neighbors Classifier.

    Returns:
        model {sklearn estimator} -- Instanciated model.
        search_space {dict} -- Search interval for each parameter of model to optimize.
        n_iters {int} -- Number of iterations allowed during optimization.
    """
    search_space = {'n_neighbors': (3, 30),
                    'weights': ['uniform', 'distance']}
    model = KNeighborsClassifier()
    n_iters = 15
    return model, search_space, n_iters


def get_MPC():
    """ Instanciate Multi-layer Perceptron Classifier.

    Returns:
        model {sklearn estimator} -- Instanciated model.
        search_space {dict} -- Search interval for each parameter of model to optimize.
        n_iters {int} -- Number of iterations allowed during optimization.
    """
    search_space = {'alpha': (1e-4, 1e-1, 'log-uniform'),
                    'batch_size': [32, 'auto'],
                    'hidden_layer_sizes': [(100), (150), (300)]}
    default_params = {'learning_rate': 'adaptive',
                      'max_iter': 200,
                      'early_stopping': True}
    model = MLPClassifier(**default_params, random_state=SEED)
    n_iters = 15
    return model, search_space, n_iters


def get_VC(estimators, weights=None):
    """ Instanciate K-Nearest Neighbors Classifier.

    Arguments:
        estimators {list[sklearn estimator]} -- List of optimized base models.

    Keyword Arguments:
        weights {list[float]} -- Associate a weight to each model's prediction (default: {None})

    Returns:
        model {sklearn estimator} -- Instanciated model.
        search_space {dict} -- Search interval for each parameter of model to optimize.
    """
    search_space = {}
    model = VotingClassifier(estimators, weights=weights)
    return model, search_space


def compute_weights(base_models):
    """Compute weights associated to each model based on their order.

    Arguments:
        base_models {list[sklearn estimator]} -- Ordered list of base models, from best to worse.

    Returns:
        weights -- Normalized weights associated to each base model.
    """
    weights = np.arange(len(base_models), 0, -1)
    return weights / np.sum(weights)
