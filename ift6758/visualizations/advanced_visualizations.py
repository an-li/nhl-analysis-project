import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from ift6758.utilities.game_utilities import filter_by_team_and_season, generate_shot_map_matrix, get_goals_per_game
from ift6758.utilities.math_utilities import subtract_and_align_matrices


def generate_static_shot_map(plays_df: pd.DataFrame, team: str, season: int, save: bool = True,
                             plot: bool = True, path_to_save: str = "./", group_feet: int = 5):
    """
    Generate a static shot contour map with number of shots and goals scored for a specific team and season

    Args:
        plays_df: Data frame of plays, all seasons and teams combined
        team: Team name
        season: Season ID (e.g., 20162017)
        plot: Boolean to choose to plot or not
        save: Boolean to choose to save the figure
        path_to_save: path where to save the figure
        group_feet: Integer to regroup distances. For exemple if group_feet = 5, shots will be grouped 5 feet by 5 feet

    Returns:

    """

    fig = go.Figure()

    global_df = filter_by_team_and_season(plays_df, season_filter=season)
    global_matrix, global_shots_per_game, global_goals_per_game = _get_matrix_and_stats_per_game(global_df, group_feet)

    team_df = filter_by_team_and_season(plays_df, team, season)
    team_matrix, team_shots_per_game, team_goals_per_game = _get_matrix_and_stats_per_game(team_df, group_feet)

    difference_matrix = subtract_and_align_matrices(team_matrix, global_matrix, 0.0)
    # Sort by decreasing distance from center ice so center ice appears at the bottom of the graph
    difference_matrix.sort_index(ascending=False, kind='mergesort', inplace=True)

    fig.add_trace(create_contour_map(difference_matrix, team, True))

    update_figure_layout(fig, _create_title(team, season, global_shots_per_game, global_goals_per_game,
                                            team_shots_per_game, team_goals_per_game))

    if plot:
        fig.show()
    if save:
        with open(path_to_save + f'static_shot_map_{team.replace(" ", "")}_{season}.html', 'wb') as f:
            f.write(fig.to_html().encode('UTF-8'))


def generate_interactive_shot_map(plays_df: pd.DataFrame, season: int, save: bool = True, plot: bool = True,
                                  path_to_save: str = "./", group_feet: int = 5):
    """
    Generate an interactive shot contour map with number of shots and goals scored for one season, in which the user can select the team to view data for, then exports the graph as HTML

    Args:
        plays_df: Data frame of plays
        season: Season ID (e.g., 20162017)
        plot: Boolean to choose to plot or not
        save: Boolean to choose to save the figure
        path_to_save: path where to save the figure
        group_feet: Integer to regroup distances. For exemple if group_feet = 5, shots will be grouped 5 feet by 5 feet

    Returns:

    """

    fig = go.Figure()

    team_list = plays_df['team'].sort_values(kind='mergesort').unique()
    buttons = []
    visible = [False] * len(team_list)
    index = 0

    buttons.append(dict(label='Sélectionner une équipe...',
                        method='update',
                        args=[{'visible': False},
                              {'title': f'Plan des tirs en fonction de l\'équipe pour la saison {season}',
                               'showlegend': False}]))

    global_df = filter_by_team_and_season(plays_df, season_filter=season)
    global_matrix, global_shots_per_game, global_goals_per_game = _get_matrix_and_stats_per_game(global_df,
                                                                                                 group_feet)

    for current_team in team_list:
        team_df = filter_by_team_and_season(plays_df, current_team, season)
        if len(team_df) > 0:
            team_matrix, team_shots_per_game, team_goals_per_game = _get_matrix_and_stats_per_game(team_df,
                                                                                                   group_feet)

            difference_matrix = subtract_and_align_matrices(team_matrix, global_matrix, 0.0)
            # Sort by decreasing distance from center ice so center ice appears at the bottom of the graph
            difference_matrix.sort_index(ascending=False, kind='mergesort', inplace=True)

            fig.add_trace(create_contour_map(difference_matrix, current_team, False))

            visible_copy = visible.copy()
            visible_copy[index] = True

            buttons.append(dict(label=f'{current_team}',
                                method='update',
                                args=[{'visible': visible_copy},
                                      {
                                          'title': _create_title(current_team, season,
                                                                 global_shots_per_game, global_goals_per_game,
                                                                 team_shots_per_game, team_goals_per_game),
                                          'showlegend': False}]))

            index += 1

    update_figure_layout(fig, f'Plan des tirs en fonction de l\'équipe pour la saison {season}')

    create_dropdown(fig, buttons)

    if plot:
        fig.show()
    if save:
        with open(path_to_save + f'interactive_shot_map_{season}.html', 'wb') as f:
            f.write(fig.to_html().encode('UTF-8'))


def _get_matrix_and_stats_per_game(plays_df: pd.DataFrame, group_feet: int) -> (pd.DataFrame, float, float):
    """
    Get shot matrix and number of shots and goals per game

    Args:
        plays_df: Data frame of plays for a specific season and/or team
        group_feet: Integer to regroup distances

    Returns:
        Tuple containing:
            matrix: Shot matrix
            shots_per_game: Number of shots per game
            goals_per_game: Number of goals per game

    """

    matrix = generate_shot_map_matrix(plays_df, bin_size=group_feet)
    shots_per_game = matrix.sum().sum()
    goals_per_game = get_goals_per_game(plays_df)

    return matrix, shots_per_game, goals_per_game


def _create_title(team: str, season: int, global_shots_per_game: float, global_goals_per_game: float,
                  team_shots_per_game: float, team_goals_per_game: float) -> str:
    """
    Create title with team name and season, as well as number of shots and goals by hour by team and compared to NHL average
    Assume each game is one hour long

    Args:
        team: Name of team
        season: Season
        global_shots_per_game: NHL average number of shots per game of season
        global_goals_per_game: NHL average number of goals scored per game of season
        team_shots_per_game: Team average number of shots per game of season
        team_goals_per_game: Team average number of goals scored per game of season

    Returns:
        Title containing team name and season, as well as number of shots and goals by hour by team and compared to NHL average
    """

    return f'Plan des tirs de l\'équipe {team} pour la saison {season}<br>{round(team_shots_per_game, 2)} tirs par heure ({round((team_shots_per_game - global_shots_per_game) / global_shots_per_game * 100, 2)}% par rapport à la moyenne)<br>{round(team_goals_per_game, 2)} buts par heure ({round((team_goals_per_game - global_goals_per_game) / global_goals_per_game * 100, 2)}% par rapport à la moyenne)'


def create_contour_map(difference_matrix: pd.DataFrame, name: str, visible: bool = True) -> go.Contour:
    """
    Create a contour map with the difference matrix, can optionally make it invisible

    Args:
        difference_matrix: Matrix of differences between global data and team data
        name: Name of series
        visible: True to make it visible by default, False otherwise

    Returns:
        Contour map with the difference matrix
    """

    largest_diff = max(abs(difference_matrix.min().min()), abs(difference_matrix.max().max()))

    return go.Contour(name=name, z=difference_matrix, showscale=True, connectgaps=True,
                      colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1,
                                                                                   'rgb(255,0,0)']],
                      x=difference_matrix.columns, y=difference_matrix.index, zmin=-1 * largest_diff,
                      zmax=largest_diff, line_smoothing=1.3, visible=visible)


def update_figure_layout(fig, title: str):
    """
    Update layout for plot with title

    Args:
        fig: Figure
        title: Title of plot

    Returns:

    """

    fig.update_layout(
        width=800,
        height=1050,
        autosize=False,
        margin=dict(t=230, b=0, l=0, r=0),
        title=title,
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


def create_dropdown(fig, buttons):
    """
    Create dropdown for figure

    Args:
        fig: Figure
        buttons: List of buttons to add to dropdown

    Returns:

    """

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
