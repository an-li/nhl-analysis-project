from datetime import datetime

import numpy as np
import pandas as pd

from ift6758.utilities.game_utilities import plays_to_frame, get_game_data, extract_players
from ift6758.utilities.math_utilities import two_dimensional_euclidean_distance, get_angle_with_x_axis
from ift6758.utilities.schedule_utilities import get_game_list_for_date_range


def extract_and_cleanup_play_data(start_date: datetime, end_date: datetime, event_types: list = [],
                                  columns_to_keep: list = []) -> pd.DataFrame:
    """
    Extract and cleanup NHL game play-by-play data from start_date to end_date

    Args:
        start_date: Start date of query period
        end_date: End date of query period
        event_types: List of event types to filter on, or empty list for all events
        columns_to_keep: Columns of data frame to keep after cleaning the column names, except columns related to player types, or empty list to keep all columns
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

    all_plays_df.loc[all_plays_df['teamType'] == 'home', 'rinkSide'] = all_plays_df['home.rinkSide']
    all_plays_df.loc[all_plays_df['teamType'] == 'away', 'rinkSide'] = all_plays_df['away.rinkSide']

    # For shootout plays, if the target is on the right, the team is on the left and vice versa
    all_plays_df.loc[
        (all_plays_df['about.periodType'] == 'SHOOTOUT') & (all_plays_df['coordinates.x'] >= 0), 'rinkSide'] = 'left'
    all_plays_df.loc[
        (all_plays_df['about.periodType'] == 'SHOOTOUT') & (all_plays_df['coordinates.x'] < 0), 'rinkSide'] = 'right'

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
    all_plays_df['angleWithGoal'] = get_angle_with_x_axis(
        np.abs(all_plays_df['goal.x'] - all_plays_df['coordinates.x']), all_plays_df['coordinates.y'])

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
        all_plays_df = all_plays_df[columns_to_keep]

    # Finally, extract players represented as a list into columns by player type if 'players' is part of columns to extract
    return extract_players(all_plays_df) if 'players' in all_plays_df.columns else all_plays_df


def add_previous_event_for_shots_and_goals(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous event information for shots and goals with a few additional stats:
    - Seconds since previous event
    - Type of previous event
    - Coordinates of previous event
    - Distance from current event to previous one
    - Shot angle from current event to previous one
    - Rebound (bool): True if previous event was a shot, False otherwise
    - Speed (distance since previous event / Seconds since previous event)
    - Shot angle change

    Args:
        plays_df: Data frame containing all plays

    Returns:
        Data frame of shots and goals with information on the previous play
    """

    # Keep only regular and overtime plays, excluding any row that does not have a rinkSide or coordinates
    plays_df = plays_df[plays_df['periodType'].isin(['REGULAR', 'OVERTIME'])].copy()
    plays_df.dropna(subset=['rinkSide', 'x', 'y'], how='any', inplace=True)
    plays_df.reset_index(drop=True, inplace=True)

    # Make a copy of only necessary columns to build previous plays
    previous_plays = plays_df[['secondsSinceStart', 'event', 'x', 'y', 'angleWithGoal']].rename(
        columns={'secondsSinceStart': 'prevSecondsSinceStart', 'event': 'prevEvent', 'x': 'prevX', 'y': 'prevY',
                 'angleWithGoal': 'prevAngleWithGoal'})
    previous_plays.index += 1

    # Left join on plays data frame to get the previous play corresponding to each event that has defined coordinates
    plays_df = plays_df.merge(previous_plays, how='left', left_index=True, right_index=True)

    # The remaining steps are only performed on shots and goals
    plays_df = plays_df[plays_df['event'].isin(['Shot', 'Goal'])]

    # Erase previous event stats for the first event of each game
    plays_df.loc[plays_df['secondsSinceStart'] == 0, previous_plays.columns] = np.nan

    # Compute time, distance and angle change from previous event
    plays_df['secondsSincePrev'] = np.subtract(plays_df['secondsSinceStart'], plays_df['prevSecondsSinceStart'])
    plays_df['distanceFromPrev'] = two_dimensional_euclidean_distance(plays_df['x'], plays_df['y'], plays_df['prevX'],
                                                                      plays_df['prevY'])

    # Initialize angle change and rebound columns according to type of previous event
    plays_df['changeOfAngleFromPrev'] = np.where(plays_df['prevEvent'] == 'Shot', np.abs(
        np.subtract(plays_df['angleWithGoal'], plays_df['prevAngleWithGoal'])), 0)
    plays_df['rebound'] = np.where(plays_df['prevEvent'] == 'Shot', True, False)

    # Computing linear and change of angle speeds requires division, when the denominator is 0, treat time change as 1 and use the distance/angle difference as change
    with np.errstate(divide='ignore', invalid='ignore'):
        plays_df['speed'] = np.true_divide(plays_df['distanceFromPrev'], plays_df['secondsSincePrev'])
        plays_df.loc[plays_df['secondsSincePrev'] == 0, 'speed'] = plays_df['distanceFromPrev']

        plays_df['speedOfChangeOfAngle'] = np.true_divide(plays_df['changeOfAngleFromPrev'],
                                                          plays_df['secondsSincePrev'])
        plays_df.loc[plays_df['secondsSincePrev'] == 0, 'changeOfAngleFromPrev'] = plays_df['changeOfAngleFromPrev']

    return plays_df
