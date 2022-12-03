from ift6758.api.nhl_api_service import get_game_live_feed
from ift6758.features.data_extractor import add_info_for_games, add_previous_event_for_shots_and_goals
from ift6758.utilities.game_utilities import plays_to_frame

columns_to_keep = ['team', 'eventIdx', 'event', 'isGoal', 'secondaryType', 'ordinalNum', 'dateTime', 'periodType',
                   'periodTime', 'periodTimeRemaining', 'secondsSinceStart', 'strength', 'emptyNet', 'x', 'y',
                   'rinkSide', 'distanceToGoal', 'angleWithGoal', 'team.away', 'goals.away', 'team.home', 'goals.home']


class GameClient:
    def __init__(self, logger):
        self.logger = logger

    def load_shots_and_last_event(self, game_id, start_timecode):
        """
        Load shots and goals data from API, as well as the last event, for a specific game

        Args:
            game_id: ID of game
            start_timecode: IF specified, returns updates for specified game ID since the given startTimecode

        Returns:
            Shots and goals data for game specified in dict format oriented by records, as well as period, time remaining and score for last event fetched until this point
        """

        current_log = f'Retrieving data for game {game_id}'
        self.logger.auto_log(current_log, is_print=True)

        plays = plays_to_frame(get_game_live_feed(game_id, start_timecode))
        if len(plays) == 0:
            current_log = 'No events available'
            self.logger.auto_log(current_log, is_print=True)
            return plays, {}

        plays = add_info_for_games(plays, columns_to_keep, [])

        last_event = plays.tail(1)[
            ['eventIdx', 'dateTime', 'ordinalNum', 'periodTimeRemaining', 'team.away', 'goals.away', 'team.home',
             'goals.home']].to_dict(orient='records')[0]

        shots_goals = add_previous_event_for_shots_and_goals(plays).drop(columns=['gameId'])
        if len(shots_goals) == 0:
            current_log = 'No shots or goals available'
            self.logger.auto_log(current_log, is_print=True)
            return shots_goals, last_event

        # Fill missing emptyNet and strength info
        shots_goals['emptyNet'] = shots_goals['emptyNet'].fillna(0)
        shots_goals['strength'] = shots_goals['strength'].fillna('Even')

        current_log = 'Game data loaded successfully'
        self.logger.auto_log(current_log, is_print=True)

        return shots_goals, last_event
