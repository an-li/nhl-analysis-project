import os

import pandas as pd
from comet_ml import Experiment


def log_dataframe_profile(df: pd.DataFrame, project_name: str, workspace: str, asset_name: str,
                          dataframe_format: str = 'csv'):
    """
    Log data frame profile to comet.ml

    Args:
        df: Data frame to log
        project_name: Name of project
        workspace: Name of workspace
        asset_name: Name of asset
        dataframe_format: Format to log data frame

    """
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name=project_name,
        workspace=workspace
    )
    experiment.log_dataframe_profile(
        dataframe=df,
        name=asset_name,
        dataframe_format=dataframe_format  # ensure you set this flag!
    )
