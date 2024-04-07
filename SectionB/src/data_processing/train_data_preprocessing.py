import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import pickle


def preprocess_and_save_data() -> None:
    """
    Load data from CSV files, preprocess features, encode target variable, and save preprocessed data.

    Returns:
        None
    """
    df_feat = pd.read_csv('../../data/features.csv')
    df_y = pd.read_csv('../../data/labels.csv')
    df_test = pd.read_csv('../../data/test.csv')

    # Drop nulls
    temp: DataFrame = pd.merge(df_feat, df_y, on='trackID').dropna()
    df_feat = temp[temp.columns[:-1]]
    df_y = temp[['trackID', 'genre']]
    del temp

    # Enriching with two more attributes
    # Count of tags
    df_feat['tags_count'] = df_feat['tags'].apply(lambda x: len(x.split(',')))
    df_test['tags_count'] = df_test['tags'].apply(lambda x: len(x.split(',')))
    # Longest tag
    df_feat['longest_tag_length'] = df_feat['tags'].apply(lambda x: max(len(tag) for tag in x.split(',')))
    df_test['longest_tag_length'] = df_test['tags'].apply(lambda x: max(len(tag) for tag in x.split(',')))

    # Drop title and tags col
    X = df_feat.drop(columns=['title', 'tags']).set_index('trackID')
    X_eval = df_test.drop(columns=['title', 'tags']).set_index('trackID')
    Y = df_y.set_index('trackID')

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform the target variable
    Y = label_encoder.fit_transform(Y.genre)

    with open('../../data/preprocessed_data/X.pickle', 'wb') as f:
        pickle.dump(X, f)
    f.close()
    with open('../../data/preprocessed_data/Y.pickle', 'wb') as f:
        pickle.dump(Y, f)
    f.close()
    with open('../../data/preprocessed_data/X_eval.pickle', 'wb') as f:
        pickle.dump(X_eval, f)
    f.close()
    with open('../../data/preprocessed_data/label_encoder.pickle', 'wb') as f:
        pickle.dump(label_encoder, f)
    f.close()


def preprocess_request_data(track_dict: BaseModel) -> pd.DataFrame:
    """
    Preprocess track data provided as a Pydantic BaseModel and return a DataFrame.

    Args:
        track_dict (BaseModel): Pydantic BaseModel containing track data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing track features.
    """
    df_tracks = basemodel_converter(track_dict)
    # Drop nulls
    df_tracks = df_tracks.dropna()

    # Enriching with two more attributes
    # Count of tags
    df_tracks['tags_count'] = df_tracks['tags'].apply(lambda x: len(x.split(',')))
    # Longest tag
    df_tracks['longest_tag_length'] = df_tracks['tags'].apply(lambda x: max(len(tag) for tag in x.split(',')))

    # Drop title and tags col
    df_tracks = df_tracks.drop(columns=['title', 'tags']).set_index('trackID')

    return df_tracks

def basemodel_converter(track_dict: BaseModel) -> pd.DataFrame:
    """
    Convert Pydantic BaseModel to DataFrame.

    Args:
        track_dict (BaseModel): Pydantic BaseModel containing track data.

    Returns:
        pd.DataFrame: DataFrame containing track data.
    """
    if isinstance(track_dict, BaseModel):
        track_dict = track_dict.dict()
    track_list = [track for track in track_dict['tracks']]
    df_tracks: DataFrame = pd.DataFrame(track_list)
    return df_tracks
