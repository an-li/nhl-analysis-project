import seaborn as sns

from ift6758.utilities.model_utilities import download_model_from_comet, load_model_from_file

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
        self.extract_model_from_file("ift6758a-a22-g3-projet", "XGBoost_KBest_25_mutual_info_classif",
                                     "1.0.0")

        current_log = 'Default model loaded'
        self.logger.auto_log(current_log, is_print=True)

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
            self.logger.auto_log(f"Model {model_name} on filesystem, loading from file", is_print=True)
        else:
            self.logger.auto_log(f"Model {model_name} not on filesystem, loading from Comet", is_print=True)
            try:
                download_model_from_comet(workspace_name, model_name.replace('_', '-'), version,
                                          output_path=output_path)
            except Exception as e:
                current_log = 'Failed downloading the model'
                self.logger.auto_log(current_log, e, is_print=True)
                if not os.path.exists(path_to_file) or not load_already_downloaded_if_error:
                    raise e

        self.loaded_model = load_model_from_file(path_to_file)
        return self.loaded_model
