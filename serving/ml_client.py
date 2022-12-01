import sys
import traceback

import seaborn as sns

sns.set()

import pickle

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


from comet_ml import API

from flask import jsonify
import os.path


def auto_log(log, app, exception=None, is_print=False):
    if (is_print):
        print(log)
        if exception:
            print(f'Exception: {str(exception)}', file=sys.stderr)
            print(f'Stack trace: {traceback.format_exc()}', file=sys.stderr)

    response_data = {'log': log}

    if exception:
        response_data['exception'] = str(exception)
        response_data['stack_trace'] = traceback.format_exc()
        app.logger.error(response_data)
    else:
        app.logger.info(response_data)

    return response_data


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
            response_data = auto_log(current_log, app, e, is_print=True)
            return jsonify(response_data), 500

        file = open(path_to_file, 'rb')
        loaded_model = pickle.load(file)
        file.close()

    current_log = 'Default model loaded'
    auto_log(current_log, app, is_print=True)

    return loaded_model
