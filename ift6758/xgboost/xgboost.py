import os

from comet_ml import Experiment
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from ift6758.utilities.model_utilities import *


def best_hyperparameters(params: dict = {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'max_depth': [2, 4, 6, 8, 10],
    'gamma': [0.0, 0.2, 0.4, 0.6, 0.8]}, method: str = "grid", n_iter: int = 10):
    """
    xgboost best hyperparameters

    Args:
        params: parameters to test by random search
        method: choose the method of search
        n_iter: number of iteration for random search

    Return :
        Xgboost model with best hyperparameters or Xgboost classifier
    """
    model = xgb.XGBClassifier(random_state=42)
    if method == "grid":
        model = GridSearchCV(model, param_grid=params, scoring='roc_auc')
    elif method == "random":
        model = RandomizedSearchCV(xgb.XGBClassifier(random_state=42), param_distributions=params, n_iter=n_iter,
                                   scoring='roc_auc', random_state=42)
    return model


def xgboost_model(df_train: pd.DataFrame, features: list, model_name: str, project_name: str, workspace: str,
                  model=xgb.XGBClassifier(random_state=42),  # hyper parameters
                  features_selection: str = None, score_func=None, k: int = 10, min_features: int = 1,
                  # features selection
                  comet: bool = True, balanced: bool = True):
    '''
    xgboost models with all features
    
    Args : 
        df_train: training set
        features: list of features 
        model_name: name of the model
        project_name: Name of project
        workspace: Name of workspace
        model: modele of xgboost classifier to use
        features_selection: choose to d
        commet: Bool to decide to register model on comet or not
        balanced: Booléan to tell the function to balanced data or not

    Return :
        Data for train and validation, dictionary of all predictions and probabilities for each model and all commet experiments
    '''

    # Filtering and balancing dataframe
    df_filtered = prepare_df(df_train, features)
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
    x, y, x_val, y_val = get_train_validation(df_filtered, f_remove, ["isGoal"], 0.2, balanced)
    kept_features = f_remove
    if features_selection == "k_best":
        x, x_val, t = select_k_best_features(score_func, x, x_val, y, k)
        df_kept_features = pd.DataFrame({'Features': f_remove,
                                         'Features selected': t.get_support()})
        kept_features = list((df_kept_features[df_kept_features['Features selected'] == True])["Features"])

    # Instanciate a logistic regression model
    clf = model
    if features_selection == "recursive":
        clf = recursive_best_features(clf, x, y, min_features)

    # train model
    clf.fit(x, y)
    pickle.dump(clf, open("./models/" + model_name + ".pkl", "wb"))

    # get hyperparameters
    params = clf.get_params

    # score model (training set)
    score_training = clf.score(x, y)

    # score model (validation set)
    score_validation = clf.score(x_val, y_val)

    # Class predictions and probabilities 
    val_preds = clf.predict(x_val)
    score_prob = clf.predict_proba(x_val)[:, 1]
    f1 = f1_score(y_val, val_preds, average="macro")
    model = {}
    model[model_name] = {"model": clf, "val_preds": val_preds, "score_prob": score_prob, "f1": f1,
                         "features": kept_features}

    if comet:
        experiment.log_model(model_name, "./models/" + model_name + ".pkl")
        experiment.log_parameters(params)
        experiment.log_metric("train_score", score_training)
        experiment.log_metric("validation_score", score_validation)
        experiment.log_metric("f1_score", f1)
        experiment.log_confusion_matrix(y_val.astype('int32'), val_preds.astype('int32'))
        experiment.end()
    return (x, y, x_val, y_val), model, experiment
