from typing import Union, List
from fastapi import FastAPI
import uvicorn
from SectionB.src.data_processing.train_data_preprocessing import preprocess_request_data
from SectionB.src.utils.dbendpoint import lookup_genre
from SectionB.src.utils.post import UserTracks
from SectionB.src.utils.load_data_to_db import create_music_db, add_entry_to_db
from SectionB.src.utils.load_trained_model import XGBModelWrapper
from typing import Dict

# Init DB
create_music_db()
# Init model
mdl = XGBModelWrapper()
mdl.load_model()
# Init web app
app = FastAPI()

features = []


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict", response_model=Dict)
async def create_post(track_dict: UserTracks):
    # take post and pass it to model predictor
    df_tracks = preprocess_request_data(track_dict)
    Y_pred = mdl.predict(df_tracks)
    add_entry_to_db(track_dict, Y_pred)
    # mdl.predict()
    return Y_pred.to_dict()


@app.get("/genre", response_model=List)
async def get_all_posts(genre: str) -> list:
    genre_list = ["folk", "soul and reggae", "punk", "dance and electronica", \
                  "metal", "pop", "classic", "pop and rock", "jazz and blues"]
    if genre in genre_list:
        tracks = lookup_genre(genre)
        result = [track[0] for track in tracks]
    else:
        result = [f"Please pick one of these genres {genre_list}"]
    return result


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
