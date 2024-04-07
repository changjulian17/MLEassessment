import pandas as pd
import SectionB.src.utils.dbendpoint as db


# create a DB
def create_music_db():
    # Load all given data with tags
    df_feat = pd.read_csv('../../data/features.csv')
    df_y = pd.read_csv('../../data/labels.csv')

    # Drop nulls
    temp = pd.merge(df_feat, df_y, on='trackID').dropna()
    df_feat = temp[temp.columns[:-1]]
    df_y = temp[['trackID', 'genre']]

    # Combine features and target
    temp = pd.merge(df_feat, df_y, on='trackID')

    db.init_db(temp)
    data = db.get_all()
    print(f"{len(data)} entries initialised in music DB with {len(data[0])} cols")


def add_entry_to_db():
    return "values"


