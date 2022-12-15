import logging
import os

import streamlit as st
import pandas as pd
import numpy as np

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


with st.sidebar:
    # TODO: Add input for the sidebar
    option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

    option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

    option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))
    

with st.container():
    # TODO: Add Game ID input
    st.text_input("Game Id", key="gameId")
    if st.button('Say hello'):
        st.write('Why hello there')

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass