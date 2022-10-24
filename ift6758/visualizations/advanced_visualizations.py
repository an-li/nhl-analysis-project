import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from ift6758.utilities.game_utilities import filter_by_team_and_season, generate_shot_map_matrix, get_goals_per_game
from ift6758.utilities.math_utilities import subtract_and_align_matrices


def generate_static_shot_map(plays_df: pd.DataFrame, team: str = None, season: int = 0, save: bool = True,
                             plot: bool = True, path_to_save: str = "./", group_feet: int = 5):
    """
    Generate a static shot contour map with number of shots and goals scored for a specific team and season

    Args:
        plays_df: Data frame of plays, all seasons and teams combined
        plot: Boolean to choose to plot or not
        save: Boolean to choose to save the figure
        path_to_save: path where to save the figure
        group_feet: Integer to regroup distances. For exemple if group_feet = 5, shots will be grouped 5 feet by 5 feet

    Returns:

    """

    global_df = filter_by_team_and_season(plays_df, season_filter=season)
    global_matrix = generate_shot_map_matrix(global_df, bin_size=group_feet)
    global_shots_per_game = global_matrix.sum().sum()
    global_goals_per_game = get_goals_per_game(global_df)

    fig = go.Figure()

    team_df = filter_by_team_and_season(plays_df, team, season)
    team_matrix = generate_shot_map_matrix(team_df, bin_size=group_feet)
    team_shots_per_game = team_matrix.sum().sum()
    team_goals_per_game = get_goals_per_game(team_df, team_filter=team, season_filter=season)

    difference_matrix = subtract_and_align_matrices(team_matrix, global_matrix, 0.0)
    # Sort by decreasing distance from center ice so center ice appears at the bottom of the graph
    difference_matrix.sort_index(ascending=False, kind='mergesort', inplace=True)

    largest_diff = max(abs(difference_matrix.min().min()), abs(difference_matrix.max().max()))

    fig.add_trace(go.Contour(z=difference_matrix, showscale=True, connectgaps=False,
                             colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1,
                                                                                          'rgb(255,0,0)']],
                             x=difference_matrix.columns, y=difference_matrix.index, zmin=-1 * largest_diff,
                             zmax=largest_diff, line_smoothing=1.3))

    # Update plot sizing
    fig.update_layout(
        width=800,
        height=1050,
        autosize=False,
        margin=dict(t=230, b=0, l=0, r=0),
        title=f'Plan des tirs de l\'équipe {team} et de la saison {season}<br>{round(team_shots_per_game, 2)} tirs par match ({round((team_shots_per_game - global_shots_per_game) / global_shots_per_game * 100, 2)}% par rapport à la moyenne)<br>{round(team_goals_per_game, 2)} buts par match ({round((team_goals_per_game - global_goals_per_game) / global_goals_per_game * 100, 2)}% par rapport à la moyenne)',
        template="plotly_white",
    )

    # Add axes title
    fig.update_xaxes(title_text='Distance du centre de la patinoire (pi)')
    fig.update_yaxes(title_text='Distance du centre de la glace au but (pi)')

    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="auto"
    )

    img = Image.open('figures/nhl_rink_offensive.png')
    fig.add_layout_image(
        dict(
            source=img,
            xref="paper", yref="paper",
            x=0.5, y=0,
            sizex=1, sizey=1,
            xanchor="center",
            yanchor="bottom",
            opacity=0.3,
        )
    )

    if plot:
        fig.show()
    if save:
        with open(path_to_save + f'static_shot_map_{team.replace(" ", "")}_{season}.html', 'wb') as f:
            f.write(fig.to_html().encode('UTF-8'))


def generate_interactive_shot_map(plays_df: pd.DataFrame, save: bool = True, plot: bool = True,
                                  path_to_save: str = "./", group_feet: int = 5):
    """
    Generate an interactive shot contour map with number of shots and goals scored, in which the user can select the team/season combination to view data for, then exports the graph as HTML

    Args:
        plays_df: Data frame of plays, all seasons and teams combined
        plot: Boolean to choose to plot or not
        save: Boolean to choose to save the figure
        path_to_save: path where to save the figure
        group_feet: Integer to regroup distances. For exemple if group_feet = 5, shots will be grouped 5 feet by 5 feet

    Returns:

    """

    fig = go.Figure()

    team_list = plays_df['team'].sort_values(kind='mergesort').unique()
    season_list = plays_df['season'].unique()
    buttons = []
    visible = [False] * len(season_list) * len(team_list)
    index = 0

    buttons.append(dict(label='Sélectionner une équipe...',
                        method='update',
                        args=[{'visible': False},
                              {'title': 'Plan des tirs en fonction de l\'équipe et de la saison sélectionnée',
                               'showlegend': False}]))

    for current_season in season_list:
        global_df = filter_by_team_and_season(plays_df, season_filter=current_season)
        global_matrix = generate_shot_map_matrix(global_df, bin_size=group_feet)
        global_shots_per_game = global_matrix.sum().sum()
        global_goals_per_game = get_goals_per_game(global_df)
        for current_team in team_list:
            team_df = filter_by_team_and_season(plays_df, current_team, current_season)
            if len(team_df) > 0:
                team_matrix = generate_shot_map_matrix(team_df, bin_size=group_feet)
                team_shots_per_game = team_matrix.sum().sum()
                team_goals_per_game = get_goals_per_game(team_df, team_filter=current_team,
                                                         season_filter=current_season)

                difference_matrix = subtract_and_align_matrices(team_matrix, global_matrix, 0.0)
                # Sort by decreasing distance from center ice so center ice appears at the bottom of the graph
                difference_matrix.sort_index(ascending=False, kind='mergesort', inplace=True)

                largest_diff = max(abs(difference_matrix.min().min()), abs(difference_matrix.max().max()))

                fig.add_trace(go.Contour(z=difference_matrix, showscale=True, connectgaps=False,
                                         colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1,
                                                                                                      'rgb(255,0,0)']],
                                         x=difference_matrix.columns, y=difference_matrix.index, zmin=-1 * largest_diff,
                                         zmax=largest_diff, line_smoothing=1.3, visible=False))

                visible_copy = visible.copy()
                visible_copy[index] = True

                buttons.append(dict(label=f'{current_season} {current_team}',
                                    method='update',
                                    args=[{'visible': visible_copy},
                                          {
                                              'title': f'Plan des tirs de l\'équipe {current_team} et de la saison {current_season}<br>{round(team_shots_per_game, 2)} tirs par match ({round((team_shots_per_game - global_shots_per_game) / global_shots_per_game * 100, 2)}% par rapport à la moyenne)<br>{round(team_goals_per_game, 2)} buts par match ({round((team_goals_per_game - global_goals_per_game) / global_goals_per_game * 100, 2)}% par rapport à la moyenne)',
                                              'showlegend': False}]))

                index += 1

    # Update plot sizing

    fig.update_layout(
        width=800,
        height=1050,
        autosize=False,
        margin=dict(t=230, b=0, l=0, r=0),
        title='Plan des tirs en fonction de l\'équipe et de la saison sélectionnée',
        template="plotly_white",
    )

    # Add axes title
    fig.update_xaxes(title_text='Distance du centre de la patinoire (pi)')
    fig.update_yaxes(title_text='Distance du centre de la glace au but (pi)')

    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="auto"
    )

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
        ]
    )

    img = Image.open('figures/nhl_rink_offensive.png')
    fig.add_layout_image(
        dict(
            source=img,
            xref="paper", yref="paper",
            x=0.5, y=0,
            sizex=1, sizey=1,
            xanchor="center",
            yanchor="bottom",
            opacity=0.3,
        )
    )

    if plot:
        fig.show()
    if save:
        with open(path_to_save + 'interactive_shot_map.html', 'wb') as f:
            f.write(fig.to_html().encode('UTF-8'))
