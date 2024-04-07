from typing import Union
from fastapi import FastAPI
import uvicorn
from SectionB.src.utils.post import UserPost
from SectionB.src.utils.load_data_to_db import create_music_db
from SectionB.src.utils.load_trained_model import XGBModelWrapper

# Init DB
create_music_db()
# Init model
mdl = XGBModelWrapper()
mdl.load_model()
#Init web app
app = FastAPI()

features = []

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/post", response_model=UserPost)
async def create_post(feature: UserPost):
    features.append(feature)
    # take post and pass it to model predictor

    return feature


@app.get("/posts", response_model=list[UserPost])
async def get_all_posts():
    return features


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
