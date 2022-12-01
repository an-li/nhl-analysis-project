import sys
import traceback

from ift6758.api.nhl_api_service import get_game_live_feed
from ift6758.features.data_extractor import add_info_for_games, add_previous_event_for_shots_and_goals
from ift6758.utilities.game_utilities import plays_to_frame
from auto_logger import AutoLogger

columns_to_keep = ['team', 'eventIdx', 'event', 'isGoal', 'secondaryType', 'ordinalNum', 'dateTime', 'periodType',
                   'periodTime', 'periodTimeRemaining', 'secondsSinceStart', 'strength', 'emptyNet', 'x', 'y',
                   'rinkSide', 'distanceToGoal', 'angleWithGoal', 'team.away', 'goals.away', 'team.home', 'goals.home']


def load_shots_and_last_event(app, game_id, start_timecode):
    """
    Load shots and goals data from API, as well as the last event, for a specific game

    Args:
        game_id: ID of game
        start_timecode: IF specified, returns updates for specified game ID since the given startTimecode

    Returns:
        Shots and goals data for game specified in dict format oriented by records, as well as period, time remaining and score for last event fetched until this point
    """

    current_log = f'Retrieving data for game {game_id}'
    AutoLogger.auto_log(current_log, app, is_print=True)

    plays = plays_to_frame(get_game_live_feed(game_id, start_timecode))
    plays = add_info_for_games(plays, columns_to_keep, [])
    shots_goals = add_previous_event_for_shots_and_goals(plays).drop(columns=['gameId'])

    # Fill missing emptyNet and strength info
    shots_goals['emptyNet'] = shots_goals['emptyNet'].fillna(0)
    shots_goals['strength'] = shots_goals['strength'].fillna('Even')

    last_event = plays.tail(1)[
        ['eventIdx', 'dateTime', 'ordinalNum', 'periodTimeRemaining', 'team.away', 'goals.away', 'team.home',
         'goals.home']].to_dict(orient='records')[0]

    current_log = 'Game data loaded successfully'
    AutoLogger.auto_log(current_log, app, is_print=True)

    return shots_goals, last_event
