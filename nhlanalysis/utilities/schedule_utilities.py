import itertools
from datetime import datetime

import pandas as pd

from nhlanalysis.api.nhl_api_service import get_schedule_by_date_range


def get_game_list_for_date_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Get list of NHL games between specific dates, all teams combined
    Calls the API and return a data frame of games between start_date and end_date

    Args:
        start_date: Start date of query period
        end_date: End date of query period

    Returns:
        Data frame of NHL games between specified dates
    """
    schedule = get_schedule_by_date_range(start_date, end_date)

    # Unchain and flatten the 'dates'->'games' component in the schedule into one flat list
    game_list = list(itertools.chain.from_iterable([date['games'] for date in schedule['dates']]))

    return pd.json_normalize(game_list)
