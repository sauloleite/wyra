# test_data_maker.py

import pytest
from unittest.mock import MagicMock, patch
from wyra import FineTuningDataMaker

def test_format_data():
    # Arrange
    data_maker = FineTuningDataMaker()
    content = "This is a test content."

    expected_formatted_content = '{"messages": [{"role": "user", "content": "This is a test content."}]}'

    # Mock the API response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = expected_formatted_content
    mock_response.choices = [mock_choice]

    with patch.object(data_maker.client.chat.completions, 'create', return_value=mock_response):
        # Act
        formatted_data = data_maker.format_data(content)

    # Assert
    assert formatted_data == expected_formatted_content
