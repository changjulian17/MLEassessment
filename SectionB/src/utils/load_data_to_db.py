import pandas as pd
from pydantic import BaseModel

import SectionB.src.utils.dbendpoint as db
from SectionB.src.data_processing.train_data_preprocessing import basemodel_converter


def create_music_db() -> None:
    """
    Creates and initializes a music database with data from CSV files.

    Returns:
        None
    """
    # Load all given data with tags
    df_feat = pd.read_csv('SectionB/data/features.csv')
    df_y = pd.read_csv('SectionB/data/labels.csv')

    # Drop nulls
    temp = pd.merge(df_feat, df_y, on='trackID').dropna()
    df_feat = temp[temp.columns[:-1]]
    df_y = temp[['trackID', 'genre']]

    # Combine features and target
    temp = pd.merge(df_feat, df_y, on='trackID')

    db.init_db(temp)
    data = db.get_all()
    print(f"{len(data)} entries initialised in music DB with {len(data[0])} cols")


def add_entry_to_db(track_dict: BaseModel, Y_pred: pd.DataFrame) -> None:
    """
    Adds a single entry to the music database.

    Args:
        track_dict (BaseModel): Pydantic BaseModel containing track data.
        Y_pred (pd.DataFrame): DataFrame containing predicted genre for each track.

    Returns:
        None
    """
    # X dataframe
    df_tracks = basemodel_converter(track_dict)
    # combine
    combined_df = pd.merge(df_tracks, Y_pred, on='trackID')
    db.insert_track(combined_df)
    return None


