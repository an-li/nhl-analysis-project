import json
import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.logger
from PIL import Image

from ift6758.clients.game_client import GameClient
from ift6758.logging.logger import Logger
from ift6758.utilities.game_utilities import generate_shot_map_matrix
from ift6758.visualizations.advanced_visualizations import create_dropdown


# Logging setup
streamlit.logger.get_logger = logging.getLogger
streamlit.logger.setup_formatter = None
streamlit.logger.update_formatter = lambda *a, **k: None
streamlit.logger.set_log_level = lambda *a, **k: None

LOG_FILE = os.environ.get("STREAMLIT_LOG", "streamlit.log")
logger = Logger(LOG_FILE, 'streamlit', logging.INFO)
streamlit.logger = logger

# Game client setup
game_client = GameClient(logger)

# Serving Flask server setup
IP = os.environ.get("SERVING_IP", "127.0.0.1")
PORT = os.environ.get("SERVING_PORT", "8080")
address = f"http://{IP}:{PORT}"


TITLE = 'NHL Game Analyzer'

st.set_page_config(
    page_title=TITLE,
    initial_sidebar_state="expanded"
)

image = Image.open('images/ready-for-the-drop.jpg')

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(image, width=500)

with col3:
    st.write("")


st.title(TITLE)

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
            if st.session_state.get(f'predictions_{game_id}') is not None:
                st.session_state[f'predictions_{game_id}'] = pd.concat(
                    [st.session_state[f'predictions_{game_id}'], shots_goals_with_predictions], ignore_index=True)
            else:
                st.session_state[f'predictions_{game_id}'] = shots_goals_with_predictions

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
    # Add input for the sidebar
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
    # Add Game ID input
    st.text_input("Game ID", key="game_id")
    if st.button('Ping Game'):
        ping_game(st.session_state.game_id)

with st.container():
    # Add Game info and predictions
    if st.session_state.get('game_id') and st.session_state.get(f'predictions_{st.session_state.game_id}') is not None:
        # Set information variables
        team_home = st.session_state.get(f'last_event_{st.session_state.game_id}')['team.home']
        team_away = st.session_state.get(f'last_event_{st.session_state.game_id}')['team.away']

        goals_home = st.session_state.get(f'last_event_{st.session_state.game_id}')['goals.home']
        goals_away = st.session_state.get(f'last_event_{st.session_state.game_id}')['goals.away']

        game = st.session_state.get('game_id')
        period = st.session_state.get(f'last_event_{st.session_state.game_id}')['ordinalNum']
        time = st.session_state.get(f'last_event_{st.session_state.game_id}')['periodTimeRemaining']

        game_data = st.session_state.get(f'predictions_{st.session_state.game_id}')

        expected_goals = game_data[['team', 'goalProbability']].groupby('team').sum().reset_index()
        expected_goals_home = expected_goals.loc[expected_goals['team'] == team_home, 'goalProbability'].values[0].round(2)
        expected_goals_away = expected_goals.loc[expected_goals['team'] == team_away, 'goalProbability'].values[0].round(2)

        #headers with game_id, team names
        st.subheader(f'Game {game}: {team_home} vs {team_away}')

        #period, time remaining
        st.write(f'Period : {period} - {time}')

        #actual goals and goals predictions
        col1, col2, col3 = st.columns(3)
        col1.metric(label= f'{team_home} (actual)', value=f'{expected_goals_home} ({goals_home})', delta=(goals_home - expected_goals_home).round(2), delta_color="normal")
        col2.metric(label= f'{team_away} (actual)', value=f'{expected_goals_away} ({goals_away})', delta=(goals_away - expected_goals_away).round(2), delta_color="normal")
    elif st.session_state.get('game_id'):
        st.subheader(f'No data available for game {st.session_state.get("game_id")}')
    else:
        st.subheader(f'Please specify a game ID above and click \'Ping game\'')

with st.container():
    # Add data used for predictions
    if st.session_state.get('game_id') and st.session_state.get(f'predictions_{st.session_state.game_id}') is not None:

        st.header("Display Data used for predictions and predictions: ")

        game_data = st.session_state.get(f'predictions_{st.session_state.game_id}')[['eventIdx', 'team', 'period', 'periodTimeRemaining'] + features_by_model[st.session_state.model]['features'] + features_by_model[st.session_state.model]['features_to_one_hot'] + ['isGoal', 'goalProbability']]
        st.dataframe(game_data)


with st.container():
    if st.session_state.get('game_id') and st.session_state.get(f'predictions_{st.session_state.game_id}') is not None:
        chart_title = f'Number of shots by ice zone (5×5 ft)'

        game_data = st.session_state.get(f'predictions_{st.session_state.game_id}')

        fig = go.Figure()

        team_list = game_data['team'].sort_values(kind='mergesort').unique()
        buttons = []
        visible = [False] * len(team_list)
        index = 0

        buttons.append(dict(label='Select a team...',
                            method='update',
                            args=[{'visible': visible},
                                  {'title': chart_title,
                                   'showlegend': False}]))

        base_matrix = pd.DataFrame(np.zeros(shape=(21, 19)), columns=range(-45, 50, 5), index=range(0, 105, 5))

        for team in team_list:
            shot_matrix = generate_shot_map_matrix(game_data[game_data['team'] == team], 5.0)
            shot_matrix_aligned = np.add(shot_matrix.align(base_matrix, fill_value=0)[0], base_matrix)

            # Make a series for each team
            fig.add_trace(go.Contour(name=team, z=shot_matrix_aligned, showscale=True, connectgaps=True,
                                     colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(255,0,0)']],
                                     x=shot_matrix_aligned.columns, y=shot_matrix_aligned.index, line_smoothing=1.3,
                                     visible=False))

            visible_copy = visible.copy()
            visible_copy[index] = True

            buttons.append(dict(label=team,
                                method='update',
                                args=[{'visible': visible_copy},
                                      {'title': chart_title,
                                       'showlegend': False}]))

            index += 1

        fig.update_layout(
            width=800,
            height=940,
            autosize=False,
            margin=dict(t=230, b=0, l=0, r=0),
            title=chart_title,
            template="plotly_white",
        )
        # Add axes title
        fig.update_xaxes(title_text='Distance from center of rink (ft)')
        fig.update_yaxes(title_text='Distance from center ice to goal (ft)')
        # Update 3D scene options
        fig.update_scenes(
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="auto"
        )
        img = Image.open('figures/nhl_rink_offensive.png')
        fig.add_layout_image(
            dict(
                source=img,
                xref="paper", yref="paper",
                x=0.5, y=0,
                sizex=1, sizey=1,
                xanchor="center",
                yanchor="bottom",
                opacity=0.3,
            )
        )

        create_dropdown(fig, buttons)

        st.plotly_chart(fig, use_container_width=True)

        st.write('Similar to the interactive shot map done in Milestone 1 comparing a team\'s shot over all games with '
                 'the NHL average, this simplified version shows the number of shots for each team for every 5×5 ft '
                 'square.\n'
                 'Please select a team in the dropdown to see the shot map for the selected team in the '
                 'currently selected game.')
