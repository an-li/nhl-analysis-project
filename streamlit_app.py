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

st.title("INSERT APP TITLE HERE")

LOG_FILE = os.environ.get("STREAMLIT_LOG", "streamlit.log")
logger = Logger(LOG_FILE, 'streamlit', logging.INFO)

game_client = GameClient(logger)


with st.sidebar:
    # TODO: Add input for the sidebar
    pass

with st.container():
    # TODO: Add Game ID input
    pass

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass