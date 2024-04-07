import subprocess
from fastapi.testclient import TestClient
from SectionB.src.utils.main import app
import json

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().key == "Hello"


def test_read_item_invalid_item_id() -> None:
    response = client.get("/items/foo")
    assert response.status_code == 422  # 422 Unprocessable Entity for invalid query parameter


def generate_curl_post_request(url, payload):
    """
        Generate a cURL POST request for a given URL and payload.

        Parameters:
            url (str): The URL to send the POST request to.
            payload (dict): The payload to include in the POST request.

        Returns:
            subprocess.CompletedProcess: Completed process object representing the execution of the cURL command.

    """
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
    "body": "anotherrr"
}
url = 'http://0.0.0.0:8000/post'
curl_request = generate_curl_post_request(url, payload)
print(curl_request)
