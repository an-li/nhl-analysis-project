"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import sklearn
import pandas as pd
import joblib
import os.path
import pickle
import os
import logging

from waitress import serve
from comet_ml import API
from pathlib import Path
from flask import Flask, jsonify, request, abort


#import ift6758

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)


@app.route('/ping')
def do_ping():
    ping = 'Ping ...'

    return 'Hello World'

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    pass


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    raise NotImplementedError("TODO: implement this endpoint")

    response = None
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    try:
        content_json = request.get_json()
    except:
        json_format_error = 'JSON file not properly formatted'
        print(json_format_error)
        response_data = {'error': json_format_error}
        return jsonify(response_data), 400


    print(content_json)
    
    workspace = content_json['workspace']
    model = content_json['model']
    version = content_json['version']

    # TODO: check to see if the model you are querying for is already downloaded
    path_to_file = "models/"+model+".sav"
    is_model_on_disk = os.path.exists(path_to_file)
    

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    loaded_model = None
    if is_model_on_disk:
        print("Model already on disk, not downloading")
        path_relative_to_file = '../'+path_to_file
        print(path_relative_to_file)
        file = open(path_relative_to_file, 'rb')
        print('ok')
        loaded_model = pickle.load(file)
        print('ok2')
        file.close()


    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        print("Model not on disk, downloading")
        api = API()
        api.download_registry_model("ift6758a-a22-g3-projet", "MLP1", "1.0.0",
                            output_path="../models/", expand=True)

        file = open(path_to_file, 'rb')

        loaded_model = pickle.load(file)

        file.close()


    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    response = {'statis': 'model download sucess'}

    #app.logger.info(response)
    return jsonify(response), 200  


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


print('running flask app')
app.run()
#serve(app, listen='*:8080')