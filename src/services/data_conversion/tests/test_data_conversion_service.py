# services/data_conversion/tests/test_data_conversion_service.py
import pytest
from data_conversion_service import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_convert_files(client):
    data = {
        'files': (open('path/to/test.mat', 'rb'),),
        'output_dir': 'output'
    }
    response = client.post('/convert', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.json['message'] == 'Files have been converted'