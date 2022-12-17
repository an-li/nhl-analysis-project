import logging

import pandas as pd
import seaborn as sns

from ift6758.utilities.model_utilities import download_model_from_comet, load_model_from_file, \
    filter_and_one_hot_encode_features

sns.set()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


import os.path


class MLClient:
    def __init__(self, logger):
        self.logger = logger

    def load_default_model(self):
        """
        Load default model with name XGBoost_KBest_25_mutual_info_classif

        Returns:
            Default model
        """
        self.extract_model_from_file("ift6758a-a22-g3-projet", "XGBoost_KBest_25_f_classif",
                                     "1.0.0")

        current_log = 'Default model loaded'
        self.logger.auto_log(current_log, logging.INFO, is_print=True)

    def extract_model_from_file(self, workspace_name, model_name, version, extension='.pkl',
                                load_already_downloaded_if_error=False):
        """
        Load model from filesystem if exists, or comet_ml if it does not exist on filesystem

        Args:
            workspace_name: Name of comet.ml workspace
            model_name: Name of model
            version: Version of model
            extension: File extension of model
            load_already_downloaded_if_error: True to load existing model on filesystem if download fails, False otherwise

        Returns:
            Model from filesystem or comet.ml
        """
        output_path = './models'
        path_to_file = output_path + '/' + model_name + extension

        if os.path.exists(path_to_file):
            self.logger.auto_log(f"Model {model_name} on filesystem, loading from file", logging.INFO, is_print=True)
        else:
            self.logger.auto_log(f"Model {model_name} not on filesystem, loading from Comet", logging.INFO,
                                 is_print=True)
            try:
                download_model_from_comet(workspace_name, model_name.replace('_', '-'), version,
                                          output_path=output_path)
            except Exception as e:
                if not os.path.exists(path_to_file) or not load_already_downloaded_if_error:
                    raise e

        self.loaded_model = load_model_from_file(path_to_file)
        self.logger.auto_log(f'Model {model_name} loaded successfully', logging.INFO, is_print=True)
        return self.loaded_model

    def predict(self, data, features, features_to_one_hot=[], one_hot_features=[]):
        """
        Predict goal probabilities using loaded model and specified features

        Args:
            data: Shots and goals data
            features: List of features to predict on
            features_to_one_hot: List of categorical features to convert to one-hot encoding
            one_hot_features: List of features that have already been converted to one-hot encoding

        Returns:
            Data frame containing event index and goal probability of each shot, excluding shots made in the shootout period
        """
        if len(data) == 0:
            return pd.DataFrame()

        # Add eventIdx to columns prior to filtering so it can be used once all the NaN's have been dropped
        filtered_data = filter_and_one_hot_encode_features(data, ['eventIdx'] + features + features_to_one_hot)

        # Use the original eventIdx of the filtered data for the predictions, it will later be used to join with the game data
        predictions = pd.DataFrame(columns=['eventIdx'], data=filtered_data['eventIdx'])

        # For missing one hot features, add a column of 0's
        for column in one_hot_features:
            if column not in filtered_data.columns:
                filtered_data[column] = 0
        filtered_data = filtered_data[features + one_hot_features]

        predictions['goalProbability'] = self.loaded_model.predict_proba(filtered_data)[:, 1]

        return predictions
