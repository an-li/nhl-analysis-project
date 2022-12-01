import seaborn as sns
from auto_logger import AutoLogger

sns.set()

import pickle

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


from comet_ml import API

from flask import jsonify
import os.path


def load_default_model(app):
    model = "XGBoost_KBest_25_mutual_info_classif"
    path_to_file = "./models/" + model + ".pkl"
    is_model_on_disk = os.path.exists(path_to_file)
    if is_model_on_disk:
        file = open(path_to_file, 'rb')
        loaded_model = pickle.load(file)
        file.close()
    else:
        try:
            api = API()
            api.download_registry_model("ift6758a-a22-g3-projet", model.replace('_', '-'), "1.0.0",
                                        output_path="./models/", expand=True)
        except Exception as e:
            current_log = 'Failed downloading the model'
            response_data = AutoLogger.auto_log(current_log, app, e, is_print=True)
            return jsonify(response_data), 500

        file = open(path_to_file, 'rb')
        loaded_model = pickle.load(file)
        file.close()

    current_log = 'Default model loaded'
    AutoLogger.auto_log(current_log, app, is_print=True)

    return loaded_model
