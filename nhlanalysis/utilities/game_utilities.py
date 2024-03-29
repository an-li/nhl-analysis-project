import os
from os.path import dirname, abspath

import numpy as np
import orjson
import pandas as pd

from nhlanalysis.api.nhl_api_service import get_game_live_feed


def get_game_data(game_id: str, save_to_json: bool = True) -> dict:
    """
    Get the game data from local JSON file if exists, or calls NHL API to retrieve it if it does not
    A game ID is composed of the following parts:
        First 4 digits: Start year of season
        Next 2 digits: Type of game (01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star)
        Next 4 digits: Game number

    Files are organized in the following directory: game_data/{Start year of season}/{Type of game}/{Game number}.json

    Args:
        game_id: Game ID
        save_to_json: True for saving game data in .json file for easier retrieval, False otherwise

    Returns:
        Live feed of game as a dict
    """
    directory = f'data/game_data/{game_id[0:4]}/{game_id[4:6]}'
    directory_path = os.path.join(dirname(dirname(abspath(__file__))), directory)
    file_name = f'{game_id[6:10]}.json'
    file_path = os.path.join(directory_path, file_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            live_feed = orjson.loads(f.read())
    else:
        live_feed = get_game_live_feed(game_id)
        if save_to_json:
            with open(file_path, 'wb') as fp:
                fp.write(orjson.dumps(live_feed))

    return live_feed


def plays_to_frame(live_data: dict) -> pd.DataFrame:
    """
    Transform all plays from live data from JSON to a data frame, with extra columns representing the game ID, season and type of game (preseason, regular, playoff or all-star)

    Args:
        live_data: Game live data as a dict

    Returns:
        Data frame representation of game data
    """

    df = pd.json_normalize(live_data['liveData']['plays']['allPlays'])
    if len(df) == 0:
        return pd.DataFrame()

    # Add game metadata
    df['gameId'] = live_data['gamePk']
    df['season'] = live_data['gameData']['game']['season']
    df['gameType'] = live_data['gameData']['game']['type']
    df['team.away'] = live_data['gameData']['teams']['away']['name']
    df['team.home'] = live_data['gameData']['teams']['home']['name']

    # Add seconds since game start, which is ((period number - 1) × 1200) + number of seconds since period start
    df['secondsSinceStart'] = np.add(df['about.periodTime'].apply(_get_number_of_seconds_since_period_start),
                                     np.multiply(np.subtract(df['about.period'], 1), 1200))

    # Reformat the dateTime to timecode format
    df['about.dateTime'] = pd.to_datetime(df['about.dateTime']).dt.strftime("%Y%m%d_%H%M%S")

    # Set the last row of data frame with final score from linescore
    df.loc[df.index[-1], 'about.goals.away'] = live_data['liveData']['linescore']['teams']['away']['goals']
    df.loc[df.index[-1], 'about.goals.home'] = live_data['liveData']['linescore']['teams']['home']['goals']

    return df


def extract_players(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract players into columns by player type and add them to the data frame representing live, play-by-play data
    Since each play type has a different set of player types, this process is done one play type at a time

    Args:
        plays_df: Data frame representing plays generated by plays_to_frame, with the prefixes 'about', 'coordinates.',
        and 'result.' removed from the column names

    Returns:
        Data frame with additional columns for each type of player implied in the play, replacing the 'players' column,
        in increasing dateTime order
    """

    distinct_play_types = set(plays_df[
                                  'event'])  # Used to be 'result.event', but the prefix has been cleaned up prior to the call of this function

    combined_plays_df = pd.concat([_extract_players_for_type(plays_df[plays_df['event'] == play_type])
                                   for play_type in distinct_play_types], ignore_index=True)

    # For goals, scorers are also shooters
    if 'scorer' in combined_plays_df.columns:
        combined_plays_df.loc[combined_plays_df['event'] == 'Goal', 'shooter'] = combined_plays_df['scorer']

    # As players have been extracted, there is no need to keep the column 'players'
    return combined_plays_df.drop(columns=['players']).reset_index(drop=True)


def filter_by_team_and_season(plays_df, team_filter: str = None, season_filter: int = 0):
    """
    Filter play data by team and season

    Args:
        plays_df: Data frame containing all plays
            Must contain columns ['gameId', 'team', 'season', 'event', 'periodType', 'rinkSide', 'x', 'y']
        team_filter: Team name to filter on, or None for all teams
        season_filter: Season ID (e.g., 20202021) to filter on, or None for all teams

    Returns:
        Data frame containing only plays for team and/or season specified
    """

    if team_filter:
        plays_df = plays_df[plays_df['team'] == team_filter]
    if season_filter:
        plays_df = plays_df[plays_df['season'] == season_filter]

    return plays_df


def get_goals_per_game(plays_df: pd.DataFrame, team_filter: str = None, season_filter: int = 0) -> float:
    """
    Get average number of goals per team, or of all teams, in one or all seasons

    Args:
        plays_df: Data frame containing all plays
            Must contain columns ['gameId', 'team', 'season', 'event', 'periodType', 'rinkSide', 'x', 'y']
        team_filter: Team name to filter on, or None for all teams
        season_filter: Season ID (e.g., 20202021) to filter on, or None for all teams

    Returns:
        Average number of goals per game (i.e., number of goals / number of games)
    """

    # Do not count shootout plays
    plays_df = plays_df[plays_df['periodType'].isin(['REGULAR', 'OVERTIME'])].copy()

    plays_df = filter_by_team_and_season(plays_df, team_filter, season_filter)

    game_team_pairs = len(plays_df.groupby(['gameId', 'team']))

    if game_team_pairs == 0:
        return 0

    return len(plays_df[plays_df['event'] == 'Goal']) / game_team_pairs


def generate_shot_map_matrix(plays_df: pd.DataFrame, bin_size: float = 1.0) -> pd.DataFrame:
    """
    Create shot map matrix of shot count per game by coordinate in the offensive zone
    Use function filter_by_team_and_season to filter data to only contain data for specific team and/or season

    Args:
        plays_df: Data frame containing all shots and goals
            Must contain columns ['gameId', 'team', 'season', 'event', 'periodType', 'rinkSide', 'x', 'y']
        bin_size: Size of each bin of coordinates in feet

    Returns:
        Matrix of shot count per game by coordinate
    """
    x_axis_label = 'Distance from center ice to goal (ft)'
    y_axis_label = 'Distance from center of rink (ft)'
    values_label = f'Number of shots per game per {bin_size ** 2} ft²'

    # Do not count shootout plays
    plays_df = plays_df[plays_df['periodType'].isin(['REGULAR', 'OVERTIME'])].copy()

    game_team_pairs = len(plays_df.groupby(['gameId', 'team'])) if 'gameId' in plays_df.columns else len(
        plays_df.groupby(['team']))

    # Transpose x and y for plays on right side of rink to the other side so all plays are on the same side
    plays_df.loc[plays_df['rinkSide'] == 'right', 'x'] = -1 * plays_df['x']
    plays_df.loc[plays_df['rinkSide'] == 'right', 'y'] = -1 * plays_df['y']

    # Reset negative 0 coordinates to 0
    plays_df.loc[plays_df['x'] == -0, 'x'] = 0
    plays_df.loc[plays_df['y'] == -0, 'y'] = 0

    # Discard all plays in which the x coordinate after transposition is larger than 0, all other plays are not part of the offensive zone
    plays_df = plays_df[plays_df['x'] >= 0]

    # Divide each coordinate by *bin_size* to aggregate by *bin_size*-ft blocks and make blocks of *bin_size*^2 sqft
    plays_df['x'] = np.round(np.divide(plays_df['x'], bin_size), 0).astype('int64')
    plays_df['y'] = np.round(np.divide(plays_df['y'], bin_size), 0).astype('int64')

    percentages_by_coordinate = plays_df[['x', 'y']].value_counts().reset_index()

    # Divide aggregated shots count by number of total number of games the team(s) are involved in
    percentages_by_coordinate[0] = np.divide(percentages_by_coordinate[0], game_team_pairs)

    # Multiply the coordinates by bin_size so each cell represents the center of each bin of coordinates
    percentages_by_coordinate['x'] = np.multiply(percentages_by_coordinate['x'], bin_size)
    percentages_by_coordinate['y'] = np.multiply(percentages_by_coordinate['y'], bin_size)

    # x is distance from center ice to goal and y is distance from center of rink (both in feet)
    percentages_by_coordinate.rename(columns={'x': x_axis_label, 'y': y_axis_label, 0: values_label}, inplace=True)

    matrix = pd.crosstab(index=percentages_by_coordinate[x_axis_label], columns=percentages_by_coordinate[y_axis_label],
                         values=percentages_by_coordinate[values_label], aggfunc='mean').fillna(0)

    return matrix


def is_in_defensive_zone(plays_df: pd.DataFrame) -> pd.Series:
    """
    A play is in the defensive zone if:
    - x < 0 and rinkSide = 'left', or
    - x > 0 and rinkSide = 'right'

    Args:
        plays_df: Data frame of plays, must include columns ['rinkSide', 'x']

    Returns:
        Series indicating whether the play is in the defensive zone or not
    """
    return (((plays_df['rinkSide'] == 'left') & (plays_df['x'] < 0)) | (
            (plays_df['rinkSide'] == 'right') & (plays_df['x'] > 0)))


def get_non_empty_net_non_shootout_goals(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return list of non-empty net goals that are not in the shootout period

    Args:
        plays_df: Data frame of plays

    Returns:

    """
    return plays_df[
        (plays_df['event'] == 'Goal') & (plays_df['emptyNet'] != True) & (plays_df['periodType'] != 'SHOOTOUT')]


def get_fraction_of_plays_in_defensive_zone_by_game_and_team(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get fraction of plays in defensive zone by game and team

    Args:
        plays_df: Data frame of plays with column isDefensiveZone added from function is_in_defensive_zone

    Returns:
        Fraction of plays in defensive zone by gameId and team
    """
    return plays_df[['gameId', 'team', 'isDefensiveZone']].groupby(['gameId', 'team']).mean().reset_index()


def _get_number_of_seconds_since_period_start(time_string: str) -> int:
    """
    Get number of seconds since period start

    Args:
        time_string: String representation of elapsed time since beginning of game in mm:ss

    Returns:
        Number of seconds since period start
    """
    return int(time_string.split(':')[0]) * 60 + int(time_string.split(':')[1])


def _extract_players_for_type(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract players into columns by player type and add them to the data frame representing live, play-by-play data

    Args:
        plays_df: Data frame containing all play-by-play data for one specific type

    Returns:
        Data frame with additional columns for each type of player implied in the play
    """

    if len(plays_df['event'].unique()) != 1:
        raise ValueError('Play data may only contain one type of play at a time!')

    if plays_df['players'].isna().any():
        # If no players for this type of play, do not do anything
        return plays_df
    else:
        # Here, assume the first row will always have data for players, given that the input data frame only contains one type of play
        distinct_player_types = set([player['playerType'] for player in plays_df['players'].iloc[0]])

        # Extract players series into columns of players' names
        players_df = plays_df['players'].apply(_extract_player_full_names, args=(distinct_player_types,))

        # When there are no players in a category, they are denoted with an empty string, replace them with nan instead
        players_df.replace('', np.nan, inplace=True)

        # Columns in players_df are in the order of distinct_player_types, converted to lowercase for consistency
        players_df.columns = [x.lower() for x in distinct_player_types]

        # Left join on plays_df
        return plays_df.merge(players_df, left_index=True, right_index=True)


def _extract_player_full_names(players: list, distinct_player_types: set) -> pd.Series:
    """
    Extract players' full names into a series of names in which each entry is a list of players joined by ', ' of each
    specific type in distinct_player_types

    Args:
        players: List of players
        distinct_player_types: Set of distinct player types

    Returns:
        Series of players' full names in the order of distinct_player_types
    """

    return pd.Series(
        [', '.join([player['player']['fullName'] for player in players if player['playerType'] in player_type]) for
         player_type in distinct_player_types])
