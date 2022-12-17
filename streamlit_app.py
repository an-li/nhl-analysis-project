import json
import logging
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.logger

from ift6758.clients.game_client import GameClient
from ift6758.logging.logger import Logger

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

streamlit.logger.get_logger = logging.getLogger
streamlit.logger.setup_formatter = None
streamlit.logger.update_formatter = lambda *a, **k: None
streamlit.logger.set_log_level = lambda *a, **k: None

st.title("IFT6758-A22-G3-Projet")

LOG_FILE = os.environ.get("STREAMLIT_LOG", "streamlit.log")
logger = Logger(LOG_FILE, 'streamlit', logging.INFO)
streamlit.logger = logger

game_client = GameClient(logger)

PORT = "8080"
IP = "http://127.0.0.1:"
address = IP + PORT

# Save name of default model
if not st.session_state.get('model'):
    st.session_state.model = 'XGBoost_KBest_25_f_classif'

with open('features_by_model.json', 'rb') as fp:
    features_by_model = json.load(fp)


def ping_game(game_id):
    logger.auto_log(f'Predicting data for game {game_id} using model {st.session_state.model}', logging.INFO,
                    is_print=True)
    shots_goals, last_event = game_client.load_shots_and_last_event(game_id)

    # If some data already exists for game, only need to predict remaining indices
    if st.session_state.get(f'last_event_{game_id}'):
        shots_goals = shots_goals[shots_goals['eventIdx'] > st.session_state.get(f'last_event_{game_id}')['eventIdx']]

    # Perform predictions when there is at least one game event
    if len(shots_goals) > 0:
        r = requests.post(
            address + "/predict",
            json={
                'features': features_by_model[st.session_state.model]['features'],
                'features_to_one_hot': features_by_model[st.session_state.model]['features_to_one_hot'],
                'one_hot_features': features_by_model[st.session_state.model]['one_hot_features'],
                'data': shots_goals.replace({np.nan: None}).to_dict(orient='records')
            }
        )
        if r.status_code == 200:
            predictions = pd.DataFrame(r.json()['predictions'])
            shots_goals_with_predictions = shots_goals.merge(predictions, how='left', on='eventIdx')
            if st.session_state.get(f'predictions_{game_id}'):
                st.session_state[f'predictions_{game_id}'] = pd.concat(
                    [st.session_state[f'predictions_{game_id}'], shots_goals_with_predictions], ignore_index=True)
            else:
                st.session_state[f'predictions_{game_id}'] = shots_goals_with_predictions
            print(shots_goals_with_predictions.to_dict(orient='records'))

    # Finally, save last event for current game
    if last_event:
        st.session_state[f'last_event_{game_id}'] = last_event


def ping_server():
    r = requests.get(
        address + "/ping")
    print(r)


def download_model(workspace, model, version):
    r = requests.post(
        address + "/download_registry_model",
        json={
            'workspace': workspace,
            'model': model,
            'version': version
        }
    )
    if r.status_code == 200:
        # When the model is changed, clear the session state
        for key in st.session_state.keys():
            del st.session_state[key]

        st.session_state.model = model
        logger.auto_log(f'Model {model} loaded successfully', logging.INFO, is_print=True)


with st.sidebar:
    # TODO: Add input for the sidebar
    workspace_option = st.selectbox(
        'Workspace',
        ('ift6758a-a22-g3-projet',))

    model_option = st.selectbox(
        'Model',
        ('XGBoost_KBest_25_f_classif', 'XGBoost_All_Features', 'XGBoost_KBest_25_mutual_info_classif'))

    version_option = st.selectbox(
        'Version',
        ('1.0.0',))

    if st.button('Get Model'):
        download_model(workspace_option, model_option, version_option)

    # TODO: Remove this before handing the project
    if st.button('DEBUG ONLY: Test Ping Server'):
        ping_server()

with st.container():
    # TODO: Add Game ID input
    st.text_input("Game Id", key="game_id")
    if st.button('Ping Game'):
        ping_game(st.session_state.game_id)

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass
