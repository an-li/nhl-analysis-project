import numpy as np
import pandas as pd


def generate_shot_map_matrix(plays_df: pd.DataFrame, bin_size: float = 1.0, team_filter: str = None,
                             season_filter: int = 0):
    plays_df.dropna(subset=['rinkSide', 'x', 'y'], how='any', inplace=True)

    # Do not count shootout plays
    plays_df = plays_df[plays_df['periodType'].isin(['REGULAR', 'OVERTIME'])]

    if team_filter:
        plays_df = plays_df[plays_df['team'] == team_filter]

    if season_filter:
        plays_df = plays_df[plays_df['season'] == season_filter]

    number_of_games_per_team = len(plays_df['gameId'].unique())
    if not team_filter:
        # For each game, shots from both teams are included, so when there is no team filter, assume half of the shots are by each team on average (i.e., multiply denominator by 2)
        number_of_games_per_team *= 2

    # Transpose x and y for plays on right side of rink to the other side so all plays are on the same side
    plays_df.loc[plays_df['rinkSide'] == 'right', 'x'] = -1 * plays_df['x']
    plays_df.loc[plays_df['rinkSide'] == 'right', 'y'] = -1 * plays_df['y']

    # Reset negative 0 coordinates to 0
    plays_df.loc[plays_df['x'] == -0, 'x'] = 0
    plays_df.loc[plays_df['y'] == -0, 'y'] = 0

    # Discard all plays in which the x coordinate after transposition is larger than 0, all other plays are not part of the offensive zone
    plays_df = plays_df[plays_df['x'] >= 0]

    # Divide each coordinate by *bin_size* to aggregate by *bin_size*-ft blocks and make blocks of *bin_size*^2 sqft
    plays_df['x'] = np.round(np.divide(plays_df['x'], bin_size), 0).astype('int64')
    plays_df['y'] = np.round(np.divide(plays_df['y'], bin_size), 0).astype('int64')

    percentages_by_coordinate = plays_df[['x', 'y']].value_counts().reset_index()

    # Divide aggregated shots count by number of games per team
    percentages_by_coordinate[0] = np.divide(percentages_by_coordinate[0], number_of_games_per_team)

    # Multiply the coordinates by bin_size so each cell represents the center of each bin of coordinates
    percentages_by_coordinate['x'] = np.multiply(percentages_by_coordinate['x'], bin_size)
    percentages_by_coordinate['y'] = np.multiply(percentages_by_coordinate['y'], bin_size)

    # x is distance from center ice to goal and y is distance from center of rink (both in feet)
    percentages_by_coordinate.rename(
        columns={'x': 'Distance from center ice to goal (ft)', 'y': 'Distance from center of rink (ft)'}, inplace=True)

    matrix = pd.crosstab(index=percentages_by_coordinate['Distance from center ice to goal (ft)'],
                         columns=percentages_by_coordinate['Distance from center of rink (ft)'],
                         values=percentages_by_coordinate[0], aggfunc='mean').fillna(0)

    return matrix
