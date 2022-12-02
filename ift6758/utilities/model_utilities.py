import pickle
import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from comet_ml import API
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.model_selection import train_test_split

sns.set()


def prepare_df(df: pd.DataFrame, features: list, game_type: str = None):
    """
    Prepare data frame to be use. This function permite to only keep columns that we want and to balanced data if necessary.

    Args:
    	df: Data frame for the model
    	features: List of columns names of features we want to keep
    	game_type: Game type to keep if specified, keep all games otherwise
    Returns:
    	Data frame 
    """
    df_dropped = df[df['periodType'] != 'SHOOTOUT']
    if game_type or 'gameType' not in df_dropped.columns:
        df_dropped = df_dropped[df_dropped['gameType'] == game_type]
    df_dropped.loc[:, 'strength'] = df_dropped['strength'].fillna('Even')
    return df_dropped[features]


def one_hot_encode_features(df: pd.DataFrame, features: list):
    """
    Encode features that are not usable as they are

    Args:
        df: Data frame for the model
        features: List of columns names of features we want to encode
    Returns:
        Data frame 
    """
    if features:
        dummy_object = pd.get_dummies(df[features])
        df_encoded = df.merge(dummy_object, left_index=True, right_index=True)
        return df_encoded.drop(labels=features, axis=1)
    return df


def filter_and_one_hot_encode_features(data, features, game_type=None):
    """
    Filter and one hot encode features

    Args:
        data: Original data
        features: List of features to keep
        game_type: Game type to keep if specified, keep all games otherwise

    Returns:
        Data with only specified features, with categorical features one-hot encoded
    """

    df_filtered = prepare_df(data, features, game_type)
    df_filtered.dropna(inplace=True)
    df_filtered = one_hot_encode_features(df_filtered, list(df_filtered.select_dtypes(include=['object']).columns))

    return df_filtered


def select_k_best_features(score_func, x: np.array, x_test: np.array, y: np.array, k: int = 10):
    """
    Features selection using f_classif

    Args:
        score_func: function for classification
        x: train values
        x_test: test values
        y: train labels
        k: number of features to select
    Returns:
        new_x_train and new_x_test with features
    """
    t = SelectKBest(score_func=score_func, k=k).fit(x, y)
    new_x_train = t.transform(x)
    new_x_test = t.transform(x_test)
    return new_x_train, new_x_test, t


def recursive_best_features(model, x: np.array, y: np.array, min_features: int = 1):
    """
    Recursive Features selection

    Args:
        model: model to select features on
        x: train values
        y: train labels
        min_features: minimum number of features to select
    Returns:
        new_x_train and new_x_test with features
    """
    rfecv = RFECV(estimator=model, scoring="roc_auc", min_features_to_select=min_features)
    return rfecv


def get_train_validation(df: pd.DataFrame, data_features: list, labels_features: list, val_ratio: float,
                         balanced: bool = True, sampling='under'):
    """
    Get train and validation dataset. You can choose the size of each dataset and the column for labels and data.

    Args:
    	df: Data frame for the model
    	data_features: List of columns to be use for the inputs.
    	labels_features: List of columns to be use for labels
    	val_ratio: Size of the validation dataset
    	balanced: Booléan to tell the function to balanced data or not
    	sampling: Sampling method to use (over for oversampling, under for undersampling)
    Returns:
    	x_train, y_train, x_val, y_val
    """
    train, val = train_test_split(df, test_size=val_ratio, random_state=42)
    train = train.dropna()
    val = val.dropna()

    x, y = split_data_and_labels(train, data_features, labels_features)
    if balanced:
        if sampling == 'under':
            x, y = RandomUnderSampler(random_state=42).fit_resample(x, y)
        elif sampling == 'over':
            x, y = RandomOverSampler(random_state=42).fit_resample(x, y)

    x_val, y_val = split_data_and_labels(val, data_features, labels_features)
    return x, y, x_val, y_val


def split_data_and_labels(data, data_features, labels_features):
    x = data[data_features].to_numpy().reshape(-1, len(data_features))
    y = data[labels_features].to_numpy().reshape(-1)
    return x, y


def goal_rate(labels: np.array, score_prob: np.array, bin_size: int):
    """
    Get data to make goal rate graphic ans cumulative graphic. You can choose the size of the bins of percentage.

    Args:
        labels: labels from the model
        score_prob: Probabilities of predictions
        bin_size: Size of bins of percentage
    Returns:
        rate_array: array of goal rate
        index_rate: index in percentage for each goal rate
        goal_array: number of goals for each index
        total_array: total of goals and shots for each index
    """
    rate_array = []
    index_array = list(range(0, 100, bin_size))
    total_goal = 0
    goal_array = []
    for i in range(0, 100, bin_size):
        sub_array = labels[np.logical_and(score_prob >= np.percentile(score_prob, i),
                                          score_prob < np.percentile(score_prob, i + bin_size))]
        goals = np.count_nonzero(sub_array)
        shots = sub_array.size - goals
        sub_final = goals / (shots + goals)
        goal_array.append(goals)
        total_goal = total_goal + goals
        rate_array.append(sub_final * 100)
    return rate_array, index_array, goal_array, total_goal


def compute_cumulative(goal_array: list, total_goal: list):
    """
    Get cumulative data for cumulative goal rate graphic.

    Args:
        goal_array: number of goals for each index
        total_array: total of goals and shots for each index
    Returns:
        cumlative_array
    """
    cumulative_array = []
    last_elem = 0
    for i in np.flip(goal_array):
        if total_goal != 0:
            current = i / total_goal * 100 + last_elem
        else:
            current = last_elem
        cumulative_array.append(current)

        last_elem = current
    return cumulative_array


def roc_auc_curve(y_val: np.array, models: dict, model_name: str, add_random=True, save: bool = True, plot: bool = True,
                  path_to_save: str = "./"):
    '''
    plot ROC/AUC 
    
    Args :
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - model_name: name of the model for image
        - add_random: add random classifier
        - save: Boolean to choose to save the figure
        - plot: choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''

    fig, ax = plt.subplots(figsize=[14, 8])
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    for i in models:
        # ROC curve and AUC
        fpr, tpr, threshold = metrics.roc_curve(y_val, models[i]["score_prob"])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=i + ' : AUC = %0.2f' % roc_auc)

    if add_random:
        score_prob = []
        for i in range(len(y_val)):
            score_prob.append(random.uniform(0, 1))
        score_prob = np.array(score_prob)
        # ROC curve and AUC
        fpr, tpr, threshold = metrics.roc_curve(y_val, score_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="Random Uniform" + ' : AUC = %0.2f' % roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"ROC_AUC_" + model_name + ".png")
    if save or plot:
        plt.close()


def goal_rate_curve(y_val: np.array, models: dict, model_name: str, add_random=True, save: bool = True,
                    plot: bool = True, path_to_save: str = "./"):
    '''
    plot goal rate
    
    Args : 
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - model_name: name of the model for image
        - add_random: add random classifier
        - save : Boolean to choose to save the figure
        - plot : choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''

    fig, ax = plt.subplots(figsize=[14, 8])
    for i in models:
        rate_array, index_array, goal_array, total_goal = goal_rate(y_val, models[i]["score_prob"], 5)
        ax.plot(index_array, rate_array, label=i)

    if add_random:
        score_prob = []
        for i in range(len(y_val)):
            score_prob.append(random.uniform(0, 1))
        score_prob = np.array(score_prob)
        rate_array, index_array, goal_array, total_goal = goal_rate(y_val, score_prob, 5)
        ax.plot(index_array, rate_array, label="Random Uniform")
    plt.xticks(np.arange(0, 110, 10.0))
    plt.yticks(np.arange(0, 110, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_xaxis()
    plt.legend()
    plt.title("Goal Rate")
    plt.xlabel('Shot probability model percentile', fontsize=12)
    plt.ylabel('Goals / (Shots + Goals)', fontsize=12)

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"goal_rate_" + model_name + ".png")
    if save or plot:
        plt.close()


def goal_rate_cumulative_curve(y_val: np.array, models: dict, model_name: str, add_random=True, save: bool = True,
                               plot: bool = True, path_to_save: str = "./"):
    '''
    plot goal rate cumulative
    
    Args : 
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - model_name: name of the model for image
        - add_random: add random classifier
        - save : Boolean to choose to save the figure
        - plot : choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''

    fig, ax = plt.subplots(figsize=[14, 8])
    for i in models:
        rate_array, index_array, goal_array, total_goal = goal_rate(y_val, models[i]["score_prob"], 5)
        cumulative_array = compute_cumulative(goal_array, total_goal)
        ax.plot(np.flip(index_array), cumulative_array, label=i)

    if add_random:
        score_prob = []
        for i in range(len(y_val)):
            score_prob.append(random.uniform(0, 1))
        score_prob = np.array(score_prob)
        rate_array, index_array, goal_array, total_goal = goal_rate(y_val, score_prob, 5)
        cumulative_array = compute_cumulative(goal_array, total_goal)
        ax.plot(np.flip(index_array), cumulative_array, label="Random Uniform")
    plt.xticks(np.arange(0, 110, 10.0))
    plt.yticks(np.arange(0, 110, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_xaxis()
    plt.legend()
    plt.title("Cumulative % of goal")
    plt.xlabel('Shot probability model percentile', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"goal_rate_cumul_" + model_name + ".png")
    if save or plot:
        plt.close()


def calibration(y_val: np.array, models: dict, model_name: str, add_random=True, save: bool = True, plot: bool = True,
                path_to_save: str = "./"):
    '''
    plot calibration
    
    Args : 
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - model_name: name of the model for image
        - add_random: add random classifier
        - save : Boolean to choose to save the figure
        - plot : choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''

    fig, ax = plt.subplots(figsize=[14, 8])
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    for i in models:
        prob_true, prob_pred = calibration_curve(y_val, models[i]["score_prob"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=i)

    if add_random:
        score_prob = []
        for i in range(len(y_val)):
            score_prob.append(random.uniform(0, 1))
        score_prob = np.array(score_prob)
        prob_true, prob_pred = calibration_curve(y_val, score_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label="Random Uniform")

    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Calibration')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_" + model_name + ".png")
    if save or plot:
        plt.close()


def download_model_from_comet(workspace: str, registry_name: str, version: str, output_path: str = './'):
    """
    Download model from comet.ml

    Args:
        workspace: Workspace name
        registry_name: Registry name
        version: Version number
        output_path: Path to output model

    Returns:

    """
    api = API()

    api.download_registry_model(workspace, registry_name, version, output_path=output_path, expand=True)


def load_model_from_file(path: str):
    """
    Load pickle model from file

    Args:
        path: Path to model

    Returns:
        Model loaded using pickle
    """

    file = open(path, 'rb')

    # dump information to that file
    model = pickle.load(file)

    # close the file
    file.close()

    return model
