"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import json
import logging
import os
import os.path
import pickle

from comet_ml import API
from flask import Flask, request, jsonify
from waitress import serve

from auto_logger import AutoLogger
from game_client import GameClient
from ml_client import MLClient

# import ift6758

app = Flask(__name__)

# Set up logger
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
logger = AutoLogger(app, LOG_FILE, logging.INFO)

game_client = GameClient(logger)
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
    logger.auto_log(current_log, is_print=True)

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
        response_data = logger.auto_log(json_format_error, is_print=True)
        return jsonify(response_data), 400

    count = 0
    dictionary = {}
    for i in lines:
        dictionary[str(count)] = i
        count = count + 1

    response = json.dumps(dictionary, indent=4)

    return jsonify(response)


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
        app.logger.info(json)
    except:
        json_format_error = 'JSON file not properly formatted'
        response_data = logger.auto_log(json_format_error, is_print=True)
        return jsonify(response_data), 400

    print(content_json)

    workspace = content_json['workspace']
    model = content_json['model']
    version = content_json['version']
    extension = content_json.get('extension', '.pkl')

    try:
        ml_client.loaded_model = ml_client.extract_model_from_file(workspace, model, version, extension, True)
    except Exception as e:
        message = f'Cannot load model {model}!'
        response = logger.auto_log(message, e, True)
        return jsonify(response), 500

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    response = {'status': 'model retrieval successful'}

    app.logger.info(response)
    return jsonify(response), 200


@app.route("/game_data", methods=["GET"])
def get_game_data():
    """
    Handles GET requests made to http://IP_ADDRESS:PORT/invalid_game_id_message

    Returns shots and goals data for specified game ID
    """
    try:
        game_id = request.args.get("game_id")
        if not game_id or not game_id.isnumeric() or not len(game_id) == 10:
            raise
    except:
        invalid_game_id_message = 'Invalid Game ID specified'
        response_data = logger.auto_log(invalid_game_id_message, is_print=True)
        return jsonify(response_data), 400

    try:
        diff_patch = request.args.get("start_timecode")  # Optional parameter
        game_data, last_event = game_client.load_shots_and_last_event(game_id, diff_patch)
    except Exception as e:
        cannot_load_game_data_message = 'Cannot load game data'
        response_data = logger.auto_log(cannot_load_game_data_message, e, is_print=True)
        return jsonify(response_data), 400

    output = {
        'last': last_event,
        'shots': game_data.to_dict(orient='records')
    }

    return jsonify(output), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    try:
        content_json = request.get_json()
        app.logger.info(json)
    except:
        json_format_error = 'JSON file not properly formatted'
        response_data = logger.auto_log(json_format_error, is_print=True)
        return jsonify(response_data), 400

    # TODO properly parse JSON data, once we know the format
    x_val = content_json['data']

    try:
        y_pred = ml_client.loaded_model.predict(x_val)
    except:
        current_log = 'X Data was not properly formatted'
        response_data = logger.auto_log(current_log, is_print=True)
        return jsonify(response_data), 500

    response = {"Prediction": y_pred}

    app.logger.info(response)
    return jsonify(response), 200  # response must be json serializable!


# print('Running flask app in development mode.')
# app.run()
print('Running flask app in production mode.')
serve(app, listen='*:8080')
