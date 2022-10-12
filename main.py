import os
from datetime import datetime

from ift6758.features.data_extractor import extract_and_cleanup_play_data

if __name__ == "__main__":
    start_date = datetime(2016, 10, 12)
    end_date = datetime(2021, 7, 7)
    events_to_filter = ['Shot', 'Goal']

    # Extract data for events_to_filter between start_date and end_date
    # Extracted data also includes information that may be useful for later milestones
    all_plays_df = extract_and_cleanup_play_data(start_date, end_date, events_to_filter,
                                                 ['gameId', 'season', 'gameType', 'dateTime', 'team', 'event',
                                                  'secondaryType', 'description', 'period', 'periodType', 'periodTime',
                                                  'players', 'strength', 'emptyNet', 'x', 'y', 'rinkSide',
                                                  'distanceToGoal'])

    # Extract all without event filter
    events_to_unfiltered = '*'
    all_plays_df_unfiltered = extract_and_cleanup_play_data(start_date, end_date, events_to_unfiltered,
                                                 ['gameId', 'season', 'gameType', 'dateTime', 'team', 'event',
                                                  'secondaryType', 'description', 'period', 'periodType', 'periodTime',
                                                  'players', 'strength', 'emptyNet', 'x', 'y', 'rinkSide',
                                                  'distanceToGoal'])

    # Save the extracted data into a csv file to skip the long execution time to extract data for visualizations
    outdir = 'ift6758/data/extracted'
    filename = f'{"_".join([event.lower() for event in events_to_filter]) if events_to_filter else "plays"}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    all_plays_df.to_csv(os.path.join(outdir, filename), index=False)

    # Save the unfiltered data
    all_event = 'all.csv'
    all_plays_df_unfiltered.to_csv(os.path.join(outdir, all_event), index=False)
