from datetime import datetime

import pandas as pd

from ift6758.utilities.game_utilities import plays_to_frame, get_game_data, extract_players
from ift6758.utilities.schedule_utilities import get_game_list_for_date_range


def extract_and_cleanup_play_data(start_date: datetime, end_date: datetime, event_types: list, columns_to_keep: list):
    """
    Extract and cleanup NHL game play-by-play data from start_date to end_date

    Args:
        start_date: Start date of query period
        end_date: End date of query period
        event_types: List of event types to filter on
        columns_to_keep: Columns of data frame to keep, except columns related to player types
    Returns:
        Data frame of cleaned up data between start_date and end_date

    """

    # Get schedule to retrieve the list of games from the first regular season game of the 2016-2017 season and the last playoff game of the 2020-2021 season are used
    schedule_df = get_game_list_for_date_range(start_date, end_date)

    # Keep only regular season and playoff games
    schedule_df = schedule_df[schedule_df['gameType'].isin(['R', 'P'])]

    # Extract game data from all games and combine them into one single data frame
    all_plays_df = pd.concat([plays_to_frame(get_game_data(str(game_id))) for game_id in schedule_df['gamePk']],
                             ignore_index=True)

    # If there is a filter for event types, apply it
    if event_types:
        all_plays_df = all_plays_df[all_plays_df['result.event'].isin(event_types)]

    # Clean up irrelevant portion and keep only relevant columns
    all_plays_df.columns = all_plays_df.columns.str.replace('(about|result|coordinates).', '')
    all_plays_df.columns = all_plays_df.columns.str.replace('.name', '')
    all_plays_df = all_plays_df[columns_to_keep]

    # Finally, extract players represented as a list into columns by player type
    return extract_players(all_plays_df)
