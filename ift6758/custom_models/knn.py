import os
import pickle

import torch
from comet_ml import Experiment
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.autograd import Variable
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryF1Score
from sklearn.neighbors import KNeighborsClassifier

from ift6758.utilities.model_utilities import *


def knn_model(df_train: pd.DataFrame, features: list, model_name: str, project_name: str, workspace: str,
              hyper_params: dict, comet: bool = True, balanced: bool = False):
    '''
    xgboost models with all features

    Args :
        df_train: training set
        features: list of features
        model_name: name of the model
        project_name: Name of project
        workspace: Name of workspace
        hyper_params: hyperparameters to use
        comet: Bool to decide to register model on comet or not
        balanced: Booléan to tell the function to balanced data or not

    Return :
        Data for train and validation, dictionary of all predictions and probabilities for each model and all commet experiments
    '''

    # Filtering and balancing dataframe
    df_filtered = prepare_df(df_train, features)
    df_filtered.dropna(inplace=True)
    df_filtered = one_hot_encode_features(df_filtered, list(df_filtered.select_dtypes(include=['object']).columns))

    experiment = None
    if comet:
        experiment = Experiment(
            api_key=os.environ.get('COMET_API_KEY'),
            project_name=project_name,
            workspace=workspace
        )
        experiment.set_name(model_name)
    # Define data for train and validation
    features = list(df_filtered.columns)
    f_remove = features
    f_remove.remove("isGoal")
    x_train, y_train, x_test, y_test = get_train_validation(df_filtered, f_remove, ["isGoal"], 0.10, balanced)

    x_train_t = x_train.astype('float32')
    y_train_t = y_train.astype('float32')

    x_test_t = x_test.astype('float32')
    y_test_t = y_test.astype('float32')

    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(x_train_t, y_train_t)

    y_test_pred = neigh.predict(x_test_t)

    accuracy = accuracy_score(y_test, y_test_pred)
    print(accuracy)

    f1 = f1_score(y_test, y_test_pred, average='macro')
    print(f1)

    pickle.dump(neigh, open(f"./models/{model_name}.sav", 'wb'))

    model = {}
    model[model_name] = {"model": neigh, "f1": f1}

    if comet:
        experiment.log_model(model_name, "./models/" + model_name + ".sav")
        experiment.log_parameters(hyper_params)
        experiment.log_metric("f1", f1)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_parameters(hyper_params)
        experiment.end()

    return (x_train, y_train, x_test, y_test), model, experiment
