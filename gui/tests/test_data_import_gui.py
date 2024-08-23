import pytest
import requests
from unittest.mock import patch
import tkinter as tk
from tkinter import filedialog
import os

# Mock the file dialog to return a specific folder path
@patch('tkinter.filedialog.askdirectory', side_effect=['test_files/train', 'test_files/test'])
@patch('requests.post')
def test_run_data_import(mock_post, mock_askdirectory):
    from gui.data_import_gui import run_data_import

    # Mock the response from the Flask service
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"message": "Files have been converted"}

    # Run the data import function
    run_data_import()

    # Check that the POST request was made to the correct URL
    mock_post.assert_called_once_with(
        'http://localhost:5000/convert',
        files=[('files', open('test_files/train/file1.mat', 'rb')),
               ('files', open('test_files/test/file2.mat', 'rb'))],
        data={'output_dir': os.path.join(os.getcwd(), 'output_20230101_000000')}
    )

    # Check that the response was successful
    assert mock_post.return_value.status_code == 200
    assert mock_post.return_value.json.return_value['message'] == 'Files have been converted'