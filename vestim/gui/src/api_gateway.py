import requests
import logging

class APIGateway:
    def __init__(self, backend_url="http://127.0.0.1:8001"):
        self.backend_url = backend_url
        self.logger = logging.getLogger(__name__)

    def post(self, endpoint, data=None, json=None):
        url = f"{self.backend_url}/{endpoint}"
        try:
            response = requests.post(url, data=data, json=json)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error making POST request to {url}: {e}")
            return None

    def get(self, endpoint, params=None):
        url = f"{self.backend_url}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error making GET request to {url}: {e}")
            return None