from typing import Union
from fastapi import FastAPI
import uvicorn
from SectionB.src.data_processing.train_data_preprocessing import preprocess_request_data
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


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/post", response_model=Dict)
async def create_post(track_dict: UserTracks):
    # take post and pass it to model predictor
    df_tracks = preprocess_request_data(track_dict)
    Y_pred = mdl.predict(df_tracks)
    add_entry_to_db(track_dict, Y_pred)
    # mdl.predict()
    return Y_pred.to_dict()


@app.get("/posts", response_model=list[UserTracks])
async def get_all_posts():
    return features


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
