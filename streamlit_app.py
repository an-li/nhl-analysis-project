import json
import logging
import os

import requests
import streamlit as st

from ift6758.clients.game_client import GameClient
from ift6758.logging.logger import Logger

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title("IFT6758-A22-G3-Projet")

LOG_FILE = os.environ.get("STREAMLIT_LOG", "streamlit.log")
logger = Logger(LOG_FILE, 'streamlit', logging.INFO)

game_client = GameClient(logger)

PORT = "8080"
IP = "http://127.0.0.1:"
address = IP + PORT


def ping_game(game_id):
    shots_goals, last_event = game_client.load_shots_and_last_event(game_id)
    r = requests.post(
        address + "/predict",
        json={
            'data': shots_goals.to_dict(orient='records')
        }
    )
    print(r.json())


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
    print(r)


with st.sidebar:
    # TODO: Add input for the sidebar
    workspace_option = st.selectbox(
        'Workspace',
        ('ift6758a-a22-g3-projet',))

    model_option = st.selectbox(
        'Model',
        ('XGBoost_All_Features', 'XGBoost_KBest_25_f_classif', 'XGBoost_KBest_25_mutual_info_classif'))

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
    st.text_input("Game Id", key="gameId")
    if st.button('Ping Game'):
        ping_game(st.session_state.gameId)

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass
