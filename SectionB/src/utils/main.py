from typing import Union
from fastapi import FastAPI
import uvicorn
from SectionB.src.utils.post import UserPost

app = FastAPI()
posts = []  # global var for user inputs


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/post", response_model=UserPost)
async def create_post(post: UserPost):
    posts.append(post)
    return post


@app.get("/posts", response_model=list[UserPost])
async def get_all_posts():
    return posts


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
