from datetime import datetime

import numpy as np
import pandas as pd

from nhlanalysis.utilities.game_utilities import plays_to_frame, get_game_data, extract_players
from nhlanalysis.utilities.math_utilities import two_dimensional_euclidean_distance, get_angle_with_x_axis
from nhlanalysis.utilities.schedule_utilities import get_game_list_for_date_range


def extract_and_cleanup_play_data(start_date: datetime, end_date: datetime, event_types: list = [],
                                  columns_to_keep: list = []) -> pd.DataFrame:
    """
    Extract and cleanup NHL game play-by-play data from start_date to end_date

    Args:
        start_date: Start date of query period
        end_date: End date of query period
        event_types: List of event types to filter on, or empty list for all events
        columns_to_keep: Columns of data frame to keep after cleaning the column names, except columns related to player types, or empty list to keep all columns
            Must include gameId and eventIdx
    Returns:
        Data frame of cleaned up data between start_date and end_date, with additional columns

    """
    # Get schedule to retrieve the list of games from start_date and end_date
    schedule_df = get_game_list_for_date_range(start_date, end_date)

    # Keep only regular season and playoff games
    schedule_df = schedule_df[schedule_df['gameType'].isin(['R', 'P'])]

    # Extract game data from all games and combine them into one single data frame
    all_plays_df = pd.concat([plays_to_frame(get_game_data(str(game_id))) for game_id in schedule_df['gamePk']],
                             ignore_index=True)

    all_plays_df = add_info_for_games(all_plays_df, columns_to_keep, event_types)

    # Finally, sort combined play data in increasing gameId then eventIdx order
    return all_plays_df.sort_values(by=['gameId', 'eventIdx'], kind='mergesort').reset_index(drop=True)


def add_info_for_games(all_plays_df, columns_to_keep, event_types):
    """
    Add additional info for games, and clean up redundant part of column names

    Args:
        all_plays_df: Start date of query period
        event_types: List of event types to filter on, or empty list for all events
        columns_to_keep: Columns of data frame to keep after cleaning the column names, except columns related to player types, or empty list to keep all columns
            Must include gameId and eventIdx
    Returns:
        Data frame of cleaned up data between start_date and end_date, with additional columns
    """

    # Join rinkside information with plays
    rinkside_information = _get_rinkside_information_by_game_team_and_period(all_plays_df)
    all_plays_df = all_plays_df.merge(rinkside_information, how='left', on=['gameId', 'about.period', 'team.name'])

    # Add the position of opponent's goal depending on the rink side
    # Goal line is 11 ft from center ice, or 89 ft from center ice (coordinates: (0, 0))
    # Left and right X coordinates are intentionally reversed because the team on the left shoots to the goal of their opponent on the right and vice versa
    # Y coordinates is always 0 because the goal is on vertical center
    all_plays_df.loc[all_plays_df['rinkSide'] == 'left', 'goal.x'] = 89
    all_plays_df.loc[all_plays_df['rinkSide'] == 'right', 'goal.x'] = -89
    all_plays_df.loc[~all_plays_df['rinkSide'].isna(), 'goal.y'] = 0

    # Compute the 2D Euclidean distance to the goal associated to the team's opponent depending on the side of the ice of the team
    all_plays_df['distanceToGoal'] = two_dimensional_euclidean_distance(all_plays_df['coordinates.x'],
                                                                        all_plays_df['coordinates.y'],
                                                                        all_plays_df['goal.x'], all_plays_df['goal.y'])

    # Compute angle with respect to the X-axis of the goal
    # Angle is positive for plays on the right of the net, negative for plays on the left of the net
    # Invert the direction of the subtraction when goal.x < 0 to get the positive difference between the play and the goal
    # For plays in which the goal is on the right, the y will need to be flipped as the right side of the net has negative y
    all_plays_df['angleWithGoal'] = get_angle_with_x_axis(
        np.where(all_plays_df['goal.x'] >= 0, all_plays_df['goal.x'] - all_plays_df['coordinates.x'],
                 all_plays_df['coordinates.x'] - all_plays_df['goal.x']),
        np.where(all_plays_df['goal.x'] > 0, -1 * all_plays_df['coordinates.y'], all_plays_df['coordinates.y']))

    # Normalize angles with value -0 to 0
    all_plays_df.loc[all_plays_df['angleWithGoal'] == -0.0, 'angleWithGoal'] = 0.0

    # Set isGoal depending on if it is a shot or a goal
    all_plays_df.loc[all_plays_df['result.event'] == 'Shot', 'isGoal'] = 0
    all_plays_df.loc[all_plays_df['result.event'] == 'Goal', 'isGoal'] = 1

    # For goals, if there is no empty net information, set emptyNet to False
    all_plays_df.loc[
        (all_plays_df['result.event'] == 'Goal') & (all_plays_df['result.emptyNet'].isna()), 'result.emptyNet'] = False
    all_plays_df['result.emptyNet'] = all_plays_df['result.emptyNet'].astype(float)

    # If there is a filter for event types, apply it
    if event_types:
        all_plays_df = all_plays_df[all_plays_df['result.event'].isin(event_types)]

    # Clean up redundant portion of column names
    all_plays_df.columns = all_plays_df.columns.str.replace('(about|result|coordinates).', '')
    all_plays_df.columns = all_plays_df.columns.str.replace('.name', '')

    # Drop all duplicated columns
    all_plays_df = all_plays_df.loc[:, ~all_plays_df.columns.duplicated()]

    # If specified, keep only specified columns in addition to columns related to player types extracted later
    if columns_to_keep:
        # Add eventIdx and gameId if they are not part of columns_to_keep, they will be required for sorting
        if 'eventIdx' not in columns_to_keep:
            columns_to_keep = ['eventIdx'] + columns_to_keep
        if 'gameId' not in columns_to_keep:
            columns_to_keep = ['gameId'] + columns_to_keep

        all_plays_df = all_plays_df[columns_to_keep]

    # Extract players represented as a list into columns by player type if 'players' is part of columns to extract
    if 'players' in all_plays_df.columns:
        all_plays_df = extract_players(all_plays_df)

    return all_plays_df


def _get_rinkside_information_by_game_team_and_period(all_plays_df):
    """
    Get rinkside information by game, team and period by taking the median x-coordinate of all shots and goals
    The team plays on the right for that period when the median x-coordinate of shots is less than 0, and vice versa

    Args:
        all_plays_df: Data frame of plays containing at least columns 'gameId', 'about.period', 'team.name', 'coordinates.x'

    Returns:
        Rink side information by game, period and team name
    """

    rinkside_information = all_plays_df.loc[
        all_plays_df['result.event'].isin(['Shot', 'Goal']), ['gameId', 'about.period', 'team.name',
                                                              'coordinates.x']].groupby(
        ['gameId', 'about.period', 'team.name']).median().reset_index()
    rinkside_information['rinkSide'] = np.where(rinkside_information['coordinates.x'] < 0, 'right', 'left')
    rinkside_information.drop(columns=['coordinates.x'], inplace=True)

    return rinkside_information


def add_previous_event_for_shots_and_goals(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep and add previous event information for shots and goals, excluding those in the shootout, with a few additional stats:
    - 'prevSecondsSinceStart': Seconds since previous event
    - 'prevEvent': Type of previous event
    - 'prevX' and 'prevY': Coordinates of previous event
    - 'prevAngleWithGoal': Shot angle of previous event
    - 'rebound' (bool): True if previous event was a shot by the same team, False otherwise
    - 'secondsSincePrev': Seconds since last event
    - 'distanceFromPrev': Distance between current event and previous one
    - 'speed': Speed (distance from previous event / seconds from previous event) in ft/s
    - 'changeOfAngleFromPrev': Shot angle change between current event and previous one in deg/s

    Args:
        plays_df: Data frame containing all plays

    Returns:
        Data frame of shots and goals with information on the previous play
    """

    # Exclude any row that does not have a rinkSide or coordinates
    plays_df = plays_df.copy()
    plays_df.dropna(subset=['rinkSide', 'x', 'y'], how='any', inplace=True)
    plays_df.reset_index(drop=True, inplace=True)

    # Make a copy of only necessary columns to build previous plays
    previous_plays = plays_df[['gameId', 'secondsSinceStart', 'team', 'event', 'x', 'y', 'angleWithGoal']].rename(
        columns={'gameId': 'prevGameId', 'secondsSinceStart': 'prevSecondsSinceStart', 'team': 'prevTeam',
                 'event': 'prevEvent', 'x': 'prevX', 'y': 'prevY', 'angleWithGoal': 'prevAngleWithGoal'})
    previous_plays.index += 1

    # Left join on plays data frame to get the previous play corresponding to each event that has defined coordinates
    plays_df = plays_df.merge(previous_plays, how='left', left_index=True, right_index=True)

    # Erase previous event stats for the first event of each game, as well as shootout plays
    plays_df.loc[(plays_df['gameId'] != plays_df['prevGameId']) | (
            plays_df['periodType'] == 'SHOOTOUT'), previous_plays.columns] = np.nan

    # The remaining steps are only performed on shots and goals
    plays_df = plays_df[plays_df['event'].isin(['Shot', 'Goal'])]

    # When there are no shots or goals, return
    if len(plays_df) == 0:
        return plays_df

    # Set 'secondaryType' for shots and goals as 'shotType'
    plays_df.rename(columns={'secondaryType': 'shotType'}, inplace=True)

    # Compute time, distance and angle change from previous event
    # Note that data given by trhe NHL API are at 1-second precision, events that happen within one second of each other have a time difference of 0 seconds
    plays_df['secondsSincePrev'] = np.subtract(plays_df['secondsSinceStart'], plays_df['prevSecondsSinceStart'])
    plays_df['distanceFromPrev'] = two_dimensional_euclidean_distance(plays_df['x'], plays_df['y'], plays_df['prevX'],
                                                                      plays_df['prevY'])

    # Initialize angle change and rebound columns according to type of previous event
    # A rebound is a shot by the same team immediately following another one
    plays_df['rebound'] = ((plays_df['prevEvent'] == 'Shot') & (plays_df['team'] == plays_df['prevTeam']))
    plays_df['changeOfAngleFromPrev'] = np.where(plays_df['rebound'], np.abs(
        np.subtract(plays_df['angleWithGoal'], plays_df['prevAngleWithGoal'])), 0)

    # Computing linear and change of angle speeds requires division
    # When the denominator (time) is 0, treat time change as 1 and use the distance/angle difference as change
    with np.errstate(divide='ignore', invalid='ignore'):
        plays_df['speed'] = np.true_divide(plays_df['distanceFromPrev'], plays_df['secondsSincePrev'])
        plays_df.loc[plays_df['secondsSincePrev'] == 0, 'speed'] = plays_df['distanceFromPrev']

        plays_df['speedOfChangeOfAngle'] = np.where(plays_df['rebound'],
                                                    np.true_divide(plays_df['changeOfAngleFromPrev'],
                                                                   plays_df['secondsSincePrev']), 0)
        plays_df.loc[(plays_df['secondsSincePrev'] == 0) & (plays_df['rebound']), 'speedOfChangeOfAngle'] = plays_df[
            'changeOfAngleFromPrev']

    return plays_df.drop(columns=['prevGameId', 'prevTeam'])
