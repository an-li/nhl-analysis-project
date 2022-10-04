from datetime import datetime

from ift6758.features.data_extractor import extract_and_cleanup_play_data

if __name__ == "__main__":
    start_date = datetime(2016, 10, 12)
    end_date = datetime(2021, 7, 7)
    all_plays_df = extract_and_cleanup_play_data(start_date, end_date, ['Shot', 'Goal'],
                                                 ['gameId', 'season', 'gameType', 'dateTime', 'team', 'event',
                                                  'secondaryType', 'description', 'period',
                                                  'periodType', 'periodTime', 'players', 'strength', 'emptyNet', 'x',
                                                  'y', 'rinkSide', 'distanceToGoal'])
    all_plays_df.to_csv(f'plays_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv', index=False)
