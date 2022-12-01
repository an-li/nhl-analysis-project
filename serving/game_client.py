from ift6758.api.nhl_api_service import get_game_live_feed
from ift6758.features.data_extractor import add_info_for_games, add_previous_event_for_shots_and_goals
from ift6758.utilities.game_utilities import plays_to_frame

columns_to_keep = ['team', 'eventIdx', 'event', 'isGoal', 'secondaryType', 'ordinalNum', 'periodType', 'periodTime',
                   'periodTimeRemaining', 'secondsSinceStart', 'strength', 'emptyNet', 'x', 'y', 'rinkSide',
                   'distanceToGoal', 'angleWithGoal', 'team.away', 'goals.away', 'team.home', 'goals.home']


def auto_log(log, app, is_print=False):
    if (is_print):
        print(log)
    response_data = {'log': log}
    app.logger.info(response_data)
    return response_data


def load_shots_and_goals_data(app, game_id, start_timecode):
    """
    Load shots and goals data from API for specific game

    Args:
        game_id: ID of game
        start_timecode: IF specified, returns updates for specified game ID since the given startTimecode

    Returns:
        Shots and goals data for game specified in dict format oriented by records
    """

    current_log = f'Retrieving data for game {game_id}'
    auto_log(current_log, app, is_print=True)

    plays = plays_to_frame(get_game_live_feed(game_id, start_timecode))
    plays = add_info_for_games(plays, columns_to_keep, [])
    shots_goals = add_previous_event_for_shots_and_goals(plays).drop(columns=['gameId'])

    # Fill missing emptyNet and strength info
    shots_goals['emptyNet'] = shots_goals['emptyNet'].fillna(0)
    shots_goals['strength'] = shots_goals['strength'].fillna('Even')

    return shots_goals.to_dict(orient='records')
