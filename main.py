from datetime import datetime

from ift6758.features.data_extractor import extract_and_cleanup_play_data

if __name__ == "__main__":
    all_plays_df = extract_and_cleanup_play_data(datetime(2016, 10, 12), datetime(2021, 7, 7), ['Shot', 'Goal'],
                                                 ['gameId', 'season', 'gameType', 'dateTime', 'team', 'event',
                                                  'secondaryType', 'description', 'period',
                                                  'periodType', 'periodTime', 'players', 'strength', 'emptyNet', 'x',
                                                  'y', 'rinkSide', 'distanceToGoal'])
