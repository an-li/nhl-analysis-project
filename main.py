import os
from datetime import datetime

import pandas as pd

from ift6758.features.data_extractor import extract_and_cleanup_play_data
from ift6758.visualizations.simple_visualizations import shots_efficiency_by_type, shots_efficiency_by_distance, \
    shots_efficiency_by_type_and_distance

if __name__ == "__main__":
    start_date = datetime(2016, 10, 12)
    end_date = datetime(2021, 7, 7)

    outdir = 'ift6758/data/extracted'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    filename = f'all_events_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
    path = os.path.join(outdir, filename)
    if not os.path.exists(path):
        # Extract data for all events between start_date and end_date
        # Extracted data also includes information that may be useful for later milestones
        print("Downloading/Reading...")
        all_plays_df = extract_and_cleanup_play_data(start_date, end_date,
                                                     columns_to_keep=['gameId', 'season', 'gameType', 'dateTime',
                                                                      'team', 'event', 'secondaryType', 'description',
                                                                      'period', 'periodType', 'periodTime', 'players',
                                                                      'strength', 'emptyNet', 'x', 'y', 'rinkSide',
                                                                      'distanceToGoal'])

        # Save the unfiltered data
        print("Saving all events DataFrame...")
        all_plays_df.to_csv(os.path.join(outdir, filename), index=False)
    else:
        all_plays_df = pd.read_csv(path)

    # Convert all numeric types
    all_plays_df = all_plays_df.apply(pd.to_numeric, args=('ignore',))

    # Filter out for shots and goals only
    events_to_filter = ['Shot', 'Goal']
    filename = f'{"_".join([event.lower() for event in events_to_filter]) if events_to_filter else "plays"}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
    path = os.path.join(outdir, filename)
    if not os.path.exists(os.path.join(outdir, filename)):
        all_plays_df_filtered = all_plays_df[all_plays_df['event'].isin(events_to_filter)]
        # Save the extracted data into a csv file to skip the long execution time to extract data for visualizations
        print("Saving filtered events DataFrame...")
        all_plays_df_filtered.to_csv(os.path.join(outdir, filename), index=False)
    else:
        all_plays_df_filtered = pd.read_csv(path)

    # Make and save simple visualizations in ./figures directory
    shots_efficiency_by_type(all_plays_df_filtered, 20182019, plot=False, path_to_save="./figures/")
    shots_efficiency_by_distance(all_plays_df_filtered, [20182019, 20192020, 20202021], plot=False,
                                 path_to_save="./figures/")
    shots_efficiency_by_type_and_distance(all_plays_df_filtered, 20182019, plot=False, path_to_save="./figures/")
