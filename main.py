import os
from datetime import datetime

import pandas as pd

from ift6758.features.data_extractor import extract_and_cleanup_play_data, add_previous_event_for_shots_and_goals
from ift6758.features.incorrect_feature_analysis import incorrect_feature_analysis
from ift6758.visualizations.advanced_visualizations import generate_interactive_shot_map, generate_static_shot_map
from ift6758.visualizations.simple_visualizations import shots_efficiency_by_type, shots_efficiency_by_distance, \
    shots_efficiency_by_type_and_distance

if __name__ == "__main__":
    start_date = datetime(2015, 10, 7)  # Start of 2015-2016
    end_date = datetime(2021, 7, 7)  # End of 2020-2021

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
                                                                      'team', 'eventIdx', 'event', 'isGoal',
                                                                      'secondaryType', 'description', 'period',
                                                                      'periodType', 'periodTime', 'secondsSinceStart',
                                                                      'players', 'strength', 'emptyNet', 'x', 'y',
                                                                      'rinkSide', 'distanceToGoal', 'angleWithGoal'])

        # Save the unfiltered data
        print("Saving all events DataFrame...")
        all_plays_df.to_csv(os.path.join(outdir, filename), index=False)
    else:
        all_plays_df = pd.read_csv(path)

    # Convert all numeric types
    all_plays_df = all_plays_df.apply(pd.to_numeric, args=('ignore',))

    # Filter out for shots and goals only, data frame will later be used for feature engineering
    filename = f'shot_goal_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
    path = os.path.join(outdir, filename)
    if not os.path.exists(os.path.join(outdir, filename)):
        all_plays_df_filtered = add_previous_event_for_shots_and_goals(all_plays_df)
        all_plays_df_filtered.dropna(how='all', axis=1, inplace=True)  # Drop columns that are completely null
        # Save the extracted data into a csv file to skip the long execution time to extract data for visualizations
        print("Saving filtered events DataFrame...")
        all_plays_df_filtered.to_csv(os.path.join(outdir, filename), index=False)
    else:
        all_plays_df_filtered = pd.read_csv(path)

    # Convert all numeric types
    all_plays_df_filtered = all_plays_df_filtered.apply(pd.to_numeric, args=('ignore',))

    print("Generating visualizations...")
    # Make and save simple visualizations in ./figures directory
    shots_efficiency_by_type(all_plays_df_filtered, 20182019, plot=False, path_to_save="./figures/")
    shots_efficiency_by_distance(all_plays_df_filtered, [20182019, 20192020, 20202021], plot=False,
                                 path_to_save="./figures/")
    shots_efficiency_by_type_and_distance(all_plays_df_filtered, 20182019, plot=False, path_to_save="./figures/")

    # Make and save advanced visualizations in ./figures directory
    [generate_interactive_shot_map(all_plays_df_filtered, season, plot=False, path_to_save="./figures/") for season in
     all_plays_df_filtered['season'].unique()]

    [generate_static_shot_map(all_plays_df_filtered, 'Colorado Avalanche', season, plot=False,
                              path_to_save="./figures/") for season in [20162017, 20202021]]
    [generate_static_shot_map(all_plays_df_filtered, team, season, plot=False,
                              path_to_save="./figures/") for team in ['Buffalo Sabres', 'Tampa Bay Lightning'] for
     season in [20182019, 20192020, 20202021]]

    # Create training and test data frames
    df_train = all_plays_df_filtered[
        (all_plays_df_filtered['season'].isin([20152016, 20162017, 20172018, 20182019])) & (
                all_plays_df_filtered['gameType'] == 'R') & (
                all_plays_df_filtered['periodType'] != 'SHOOTOUT')]
    df_test_regular = all_plays_df_filtered[
        (all_plays_df_filtered['season'] == 20192020) & (all_plays_df_filtered['gameType'] == 'R') & (
                all_plays_df_filtered['periodType'] != 'SHOOTOUT')]
    df_test_playoffs = all_plays_df_filtered[
        (all_plays_df_filtered['season'] == 20192020) & (all_plays_df_filtered['gameType'] == 'P')]

    # Run incorrect feature analysis and generate the relevant CSVs
    incorrect_feature_analysis(df_train)
