from fastapi.testclient import TestClient
from SectionB.src.utils.setup import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_read_item():
    response = client.get("/items/1")
    assert response.status_code == 200
    assert response.json() == {"item_id": 1, "q": None}
    print("test")


def test_read_item_with_query_param():
    response = client.get("/items/1?q=test")
    assert response.status_code == 200
    assert response.json() == {"item_id": 1, "q": "test"}


def test_read_item_invalid_item_id():
    response = client.get("/items/foo")
    assert response.status_code == 422  # 422 Unprocessable Entity for invalid query parameter
