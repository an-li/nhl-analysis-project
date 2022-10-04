from datetime import datetime

import requests

API_BASE_URL = 'https://statsapi.web.nhl.com/api/v1'


def get_schedule_by_date_range(start_date: datetime, end_date: datetime) -> dict:
    """
    Get schedule of NHL games between specific dates

    Args:
        start_date: Start date of query period
        end_date: End date of query period

    Returns:
        List of NHL games grouped by date
    """
    try:
        r = requests.get(f'{API_BASE_URL}/schedule?startDate={start_date.strftime("%Y-%m-%d")}&endDate={end_date.strftime("%Y-%m-%d")}')
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f'Cannot get schedule for dates from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}, {e}')


def get_game_live_feed(game_id: str) -> dict:
    """
    Get live feed of NHL game with specified ID

    Args:
        game_id: Game ID

    Returns:
        Live feed for NHL game
    """
    try:
        r = requests.get(f'{API_BASE_URL}/game/{game_id}/feed/live')
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f'Cannot get feed for game {game_id}, {e}')