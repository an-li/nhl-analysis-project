from datetime import datetime

import pandas as pd

from ift6758.utilities.game_utilities import plays_to_frame, get_game_data, extract_players
from ift6758.utilities.math_utilities import two_dimensional_euclidean_distance
from ift6758.utilities.schedule_utilities import get_game_list_for_date_range


def extract_and_cleanup_play_data(start_date: datetime, end_date: datetime, event_types: list = [],
                                  columns_to_keep: list = []) -> pd.DataFrame:
    """
    Extract and cleanup NHL game play-by-play data from start_date to end_date

    Args:
        start_date: Start date of query period
        end_date: End date of query period
        event_types: List of event types to filter on
        columns_to_keep: Columns of data frame to keep after cleaning the column names, except columns related to player types
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
