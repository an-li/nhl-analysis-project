"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import json
import os
import os.path

from waitress import serve

from game_client import load_shots_and_last_event
from ml_client import *

# import ift6758

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

loaded_model = None

app = Flask(__name__)


@app.route('/ping')
def do_ping():
    return 'Hello World'


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    current_log = 'Flask Application Started'
    response_data = auto_log(current_log, app, is_print=True)

    # TODO: any other initialization before the first request (e.g. load default model)
    loaded_model = load_default_model(app)


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    try:
        with open("flask.log", "r") as f:
            lines = f.readlines()
    except:
        json_format_error = 'cant read flask.log'
        response_data = auto_log(json_format_error, app, is_print=True)
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
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    try:
        content_json = request.get_json()
        app.logger.info(json)
    except:
        json_format_error = 'JSON file not properly formatted'
        response_data = auto_log(json_format_error, app, is_print=True)
        return jsonify(response_data), 400

    print(content_json)

    workspace = content_json['workspace']
    model = content_json['model']
    version = content_json['version']
    extension = content_json['extension']

    # TODO: check to see if the model you are querying for is already downloaded
    path_to_file = "./models/" + model + extension
    is_model_on_disk = os.path.exists(path_to_file)

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    if is_model_on_disk:
        current_log = 'Model already on disk, not downloading'
        response_data = auto_log(current_log, app, is_print=True)

        file = open(path_to_file, 'rb')

        loaded_model = pickle.load(file)

        file.close()


    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        current_log = 'Model not on disk, downloading'
        response_data = auto_log(current_log, app, is_print=True)

        try:
            api = API()
            api.download_registry_model(workspace, model.replace('_', '-'), version, output_path="./models/",
                                        expand=True)
        except:
            current_log = 'Failed downloading the model'
            response_data = auto_log(current_log, app, is_print=True)
            return jsonify(response_data), 500

        file = open(path_to_file, 'rb')

        loaded_model = pickle.load(file)

        file.close()

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
        response_data = auto_log(invalid_game_id_message, app, is_print=True)
        return jsonify(response_data), 400

    try:
        diff_patch = request.args.get("start_timecode")  # Optional parameter
        game_data, last_event = load_shots_and_last_event(app, game_id, diff_patch)
        current_log = 'Game data loaded successfully'
        auto_log(current_log, app, is_print=True)
    except:
        cannot_load_game_data_message = 'Cannot load game data'
        response_data = auto_log(cannot_load_game_data_message, app, is_print=True)
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
        response_data = auto_log(json_format_error, app, is_print=True)
        return jsonify(response_data), 400

    # TODO properly parse JSON data, once we know the format
    x_val = content_json['data']

    try:
        y_pred = loaded_model.predict(x_val)
    except:
        current_log = 'X Data was not properly formatted'
        response_data = auto_log(current_log, app, is_print=True)
        return jsonify(response_data), 500

    response = {"Prediction": y_pred}

    app.logger.info(response)
    return jsonify(response), 200  # response must be json serializable!


# print('Running flask app in development mode.')
# app.run()
print('Running flask app in production mode.')
serve(app, listen='*:8080')
