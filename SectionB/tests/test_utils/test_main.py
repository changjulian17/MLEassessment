import subprocess

from fastapi.testclient import TestClient
from SectionB.src.utils.main import app
import requests
import json

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


def generate_curl_post_request(url, payload):    # generates json requests for my webapp
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    curl_command = f"curl -X 'POST' '{url}' "
    for key, value in headers.items():
        curl_command += f"-H '{key}: {value}' "
    curl_command += f"-d '{json.dumps(payload)}'"

    try:
        completed_process = subprocess.run(curl_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing curl command:", e)

    return completed_process


# Example usage
payload = {
    "name": "another one",
    "body": "anotherrre"
}
url = 'http://0.0.0.0:8000/posts'
curl_request = generate_curl_post_request(url, payload)
print(curl_request)
