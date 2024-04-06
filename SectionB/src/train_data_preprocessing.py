import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


def preprocess_and_save_data():
    """
    Load data
    """
    df_feat = pd.read_csv('../data/features.csv')
    df_y = pd.read_csv('../data/labels.csv')
    df_test = pd.read_csv('../data/test.csv')

    # Drop nulls
    df_feat = df_feat.dropna()
    df_y = df_y.loc[df_feat.index]

    # Enriching with two more attributes
    # Count of tags
    df_feat['tags_count'] = df_feat['tags'].apply(lambda x: len(x.split(',')))
    df_test['tage_count'] = df_test['tags'].apply(lambda x: len(x.split(',')))
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

    with open('../data/preprocessed_data/X.pickle', 'wb') as f:
        pickle.dump(X, f)

    with open('../data/preprocessed_data/Y.pickle', 'wb') as f:
        pickle.dump(Y, f)

    with open('../data/preprocessed_data/X_eval.pickle', 'wb') as f:
        pickle.dump(X_eval, f)
