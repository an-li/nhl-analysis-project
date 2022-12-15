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

def pingGame(game_id):
    shots_goals, last_event = game_client.load_shots_and_last_event(game_id)
    r = requests.post(
        "http://127.0.0.1:8080/predict", 
        json=json.loads(shots_goals.to_json())
    )
    print(r.json())

def pingServer():
    r = requests.get(
        "http://127.0.0.1:8080/ping")


with st.sidebar:
    # TODO: Add input for the sidebar
    option = st.selectbox(
    'Workplace',
    ('Email', 'Home phone', 'Mobile phone'))

    option = st.selectbox(
    'Model',
    ('Email', 'Home phone', 'Mobile phone'))

    option = st.selectbox(
    'Version',
    ('Email', 'Home phone', 'Mobile phone'))

    if st.button('Get Model'):
        st.write('Why hello there')
    
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


