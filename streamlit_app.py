import logging
import os

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

from serving.game_client import GameClient
from serving.logger import Logger

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
address = IP+PORT

def pingGame(game_id):
    shots_goals, last_event = game_client.load_shots_and_last_event(game_id)
    r = requests.post(
        address+"/predict", 
        json=json.loads(shots_goals.to_json())
    )
    print(r.json())

def pingServer():
    r = requests.get(
        address+"/ping")

def downloadModel(workspace, model, version):
    jsonString = '{ "workspace":{workspace}, "model":{workspace}, "version":{workspace}}'.format(workspace=workspace, model=model, version=version)
    r = requests.post(
        address+"/predict", 
        json=json.loads(jsonString)
    )


with st.sidebar:
    # TODO: Add input for the sidebar
    workplaceOption = st.selectbox(
    'Workplace',
    ('ift6758a-a22-g3-projet'))

    modelOption = st.selectbox(
    'Model',
    ('XGBoost_All_Features', 'XGBoost_KBest_25_f_classif', 'XGBoost_KBest_25_mutual_info_classif'))

    versionOption = st.selectbox(
    'Version',
    ('1.0.0'))

    if st.button('Get Model'):
        downloadModel(workplaceOption, modelOption, versionOption)
    
    #TODO: Remove this before handing the project
    if st.button('DEBUG ONLY: Test Ping Server'):
        pingServer()


with st.container():
    # TODO: Add Game ID input
    st.text_input("Game Id", key="gameId")
    if st.button('Ping Game'):
        pingGame(st.session_state.gameId)

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass


