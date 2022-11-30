import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay

import matplotlib.ticker as mtick
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler

sns.set()

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay
from imblearn.over_sampling import RandomOverSampler
from comet_ml import API

import logging
from flask import Flask, jsonify, request, abort
import os.path

def auto_log(log, app, is_print=False):
    if (is_print):
        print(log)
    response_data = {'log': log}
    app.logger.info(response_data)
    return response_data

def load_default_model(app):
    model = "MLP1"
    path_to_file = "../models/"+model+".sav"
    is_model_on_disk = os.path.exists(path_to_file)
    if is_model_on_disk:
        file = open(path_to_file, 'rb')
        loaded_model = pickle.load(file)
        file.close()
    else:
        try:
            api = API()
            api.download_registry_model("ift6758a-a22-g3-projet", "MLP1", "1.0.2",
                                output_path="../models/", expand=True)
        except:
            current_log = 'Failed downloading the model'
            response_data = auto_log(current_log, app, is_print=True)
            return jsonify(response_data), 500  
        file = open(path_to_file, 'rb')
        loaded_model = pickle.load(file)
        file.close()
    return loaded_model


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(35, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.sigmoid(x)
        
        return x