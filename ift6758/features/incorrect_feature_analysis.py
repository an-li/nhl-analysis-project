import os

import pandas as pd

from ift6758.utilities.game_utilities import is_in_defensive_zone, \
    get_fraction_of_plays_in_defensive_zone_by_game_and_team


def incorrect_feature_analysis(df_train: pd.DataFrame):
    """
    Incorrect feature analysis

    Get the percentage of non-empty net goals in defensive zone, excluding those in the shootout period, and save goals in defensive zone as well as the percentage of such goals by game and team in csv files

    Args:
        df_train: Training dataset

    Returns:

    """
    outdir = 'ift6758/data/analysis'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    non_empty_net_non_shootout_goals = df_train[
        (df_train['event'] == 'Goal') & (df_train['emptyNet'] != True) & (df_train['periodType'] != 'SHOOTOUT')]
    non_empty_net_non_shootout_goals['isDefensiveZone'] = is_in_defensive_zone(non_empty_net_non_shootout_goals)
    print(
        f"Percentage of goals labeled as in defensive zone: {round(non_empty_net_non_shootout_goals['isDefensiveZone'].mean() * 100, 3)}% ({len(non_empty_net_non_shootout_goals[non_empty_net_non_shootout_goals['isDefensiveZone'] == True])}/{len(non_empty_net_non_shootout_goals)})")

    # Save goals in defensive zone to a CSV file
    non_empty_net_non_shootout_goals.loc[non_empty_net_non_shootout_goals['isDefensiveZone'] == True,
                                         ['gameId', 'team', 'eventIdx', 'periodType', 'period', 'periodTime', 'x', 'y',
                                          'rinkSide']].to_csv(
        os.path.join(outdir, 'goals_in_defensive_zone.csv'), index=False)

    # Get the fraction of goals in the defensive zone by game and team
    goals_in_defensive_zone_by_game_and_team = get_fraction_of_plays_in_defensive_zone_by_game_and_team(
        non_empty_net_non_shootout_goals)

    # Isolate aggregated goals statistics that has a percentage of isDefensiveZone greater than 0
    goals_in_defensive_zone_by_game_and_team = goals_in_defensive_zone_by_game_and_team[
        goals_in_defensive_zone_by_game_and_team['isDefensiveZone'] > 0].sort_values(by='gameId', kind='mergesort',
                                                                                     ascending=True)

    # Finally, save aggregated stats to a CSV file
    goals_in_defensive_zone_by_game_and_team.to_csv(
        os.path.join(outdir, 'nonzero_goals_in_defensive_zone_by_team_and_game.csv'), index=False)
