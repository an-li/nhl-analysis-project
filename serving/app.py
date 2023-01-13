"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import logging
import os.path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from waitress import serve

from nhlanalysis.clients.ml_client import MLClient
from nhlanalysis.logging.logger import Logger

PORT = os.environ.get("SERVING_PORT", "8080")

app = Flask(__name__)

# Set up logger
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
logger = Logger(LOG_FILE, app.name, logging.INFO)

ml_client = MLClient(logger)


@app.route('/ping')
def do_ping():
    return 'Hello World'


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """

    current_log = 'Flask Application Started'
    logger.auto_log(current_log, logging.INFO, is_print=True)

    # TODO: any other initialization before the first request (e.g. load default model)
    ml_client.load_default_model()


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    try:
        with open("flask.log", "r") as f:
            lines = f.readlines()
    except:
        json_format_error = 'cant read flask.log'
        response_data = logger.auto_log(json_format_error, logging.ERROR, is_print=True)
        return jsonify(response_data), 400

    return jsonify(lines)


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
            extension: (optional, default: .pkl)
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    try:
        content_json = request.get_json()
    except:
        json_format_error = 'JSON file not properly formatted'
        response_data = logger.auto_log(json_format_error, logging.ERROR, is_print=True)
        return jsonify(response_data), 400

    print(content_json)

    workspace = content_json['workspace']
    model = content_json['model']
    version = content_json['version']
    extension = content_json.get('extension', '.pkl')

    try:
        ml_client.extract_model_from_file(workspace, model, version, extension, True)
    except Exception as e:
        message = f'Cannot load model {model}!'
        response = logger.auto_log(message, logging.ERROR, exception=e, is_print=True)
        return jsonify(response), 500

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    response = {'status': f"Model {model} loaded successfully"}

    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    try:
        content_json = request.get_json()
    except:
        json_format_error = 'JSON file not properly formatted'
        response_data = logger.auto_log(json_format_error, logging.ERROR, is_print=True)
        return jsonify(response_data), 400

    x_val = content_json['data']
    features = content_json['features']
    features_to_one_hot = content_json.get('features_to_one_hot', [])
    one_hot_features = content_json.get('one_hot_features', [])

    try:
        predicted_data = ml_client.predict(pd.DataFrame(x_val), features, features_to_one_hot, one_hot_features)
    except Exception as e:
        current_log = 'X Data was not properly formatted'
        response_data = logger.auto_log(current_log, logging.ERROR, exception=e, is_print=True)
        return jsonify(response_data), 500

    response = {"predictions": predicted_data.replace({np.nan: None}).to_dict(orient='records')}

    logger.auto_log("Predictions loaded successfully", logging.INFO, is_print=True)
    return jsonify(response), 200  # response must be json serializable!


# print('Running flask app in development mode.')
# app.run()
print('Running flask app in production mode.')
serve(app, listen=f'*:{PORT}')
