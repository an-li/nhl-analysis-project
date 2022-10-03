from datetime import datetime

import pandas as pd

from ift6758.utilities.game_utilities import get_game_data, plays_to_frame, extract_players
from ift6758.utilities.schedule_utilities import get_game_list_for_date_range

if __name__ == "__main__":
    # Get schedule to retrieve the list of games from the first regular season game of the 2016-2017 season and the last playoff game of the 2020-2021 season are used
    schedule_df = get_game_list_for_date_range(datetime(2016, 10, 12), datetime(2021, 7, 7))

    # Keep only regular season and playoff games
    schedule_df = schedule_df[schedule_df['gameType'].isin(['R', 'P'])]

    # Extract game data from all games and combine them into one single data frame
    all_plays_df = pd.concat([plays_to_frame(get_game_data(str(game_id))) for game_id in schedule_df['gamePk']], ignore_index=True)

    # Keep only shots and goals
    all_plays_df = all_plays_df[all_plays_df['result.event'].isin(['Shot', 'Goal'])]

    # Clean up irrelevant portion and keep only relevant columns
    all_plays_df.columns = all_plays_df.columns.str.replace('(about|result|coordinates).', '')
    all_plays_df.columns = all_plays_df.columns.str.replace('.name', '')
    all_plays_df = all_plays_df[
        ['gameId', 'season', 'gameType', 'dateTime', 'team', 'event', 'secondaryType', 'description', 'period',
         'periodType', 'periodTime', 'players', 'strength', 'emptyNet', 'x', 'y']]

    # Finally, extract players represented as a list into columns by player type
    all_plays_df = extract_players(all_plays_df)

    all_plays_df