import pytest
import json
from unittest.mock import patch, MagicMock
from wyra.crypto import CryptoHandler
from wyra.data_maker import FineTuningDataMaker

# Mock CryptoHandler to avoid actual encryption/decryption during testing
@pytest.fixture
def mock_crypto_handler():
    with patch.object(CryptoHandler, '__init__', return_value=None) as mock_init, \
         patch.object(CryptoHandler, 'decrypt', return_value="tZOqzT68nu6g7INzcmNMtWZRfIMBMhMTJj3DAn66uUcfLYv/Ftcz2SW+uS6F4zmKSQBN/vBBknBJRh6LnScNabgYqq6YY/vGXsJGud82kyY=") as mock_decrypt:
        yield mock_decrypt
        mock_init.assert_called_once_with("wyra")

# Test the initialization of FineTuningDataMaker class
def test_initialization(mock_crypto_handler):
    with patch('google.generativeai.configure') as mock_configure:
        maker = FineTuningDataMaker()
        mock_crypto_handler.assert_called_once()
        mock_configure.assert_called_once_with(api_key="tZOqzT68nu6g7INzcmNMtWZRfIMBMhMTJj3DAn66uUcfLYv/Ftcz2SW+uS6F4zmKSQBN/vBBknBJRh6LnScNabgYqq6YY/vGXsJGud82kyY=")

# Test the format_data method
def test_format_data(mock_crypto_handler):
    content = "Sample content to be formatted"
    formatted_response = '{"messages": [{"role": "system", "content": "<instructions>"}]}'

    # Mock the API call to the Gemini model
    with patch('google.generativeai.GenerativeModel') as mock_model_class:
        mock_model = mock_model_class.return_value
        mock_model.generate_content = MagicMock(return_value=MagicMock(text=f'```jsonl\n{formatted_response}\n```'))
        maker = FineTuningDataMaker()
        result = maker.format_data(content)

        # Ensure that the generate_content method was called correctly
        prompt_part = "Please format the following as JSON Lines (JSONL) for fine-tuning."
        mock_model.generate_content.assert_called_once()
        assert prompt_part in mock_model.generate_content.call_args[0][0]

        # Verify the output
        assert json.loads(result) == json.loads(formatted_response)

# Test exception handling in format_data
def test_format_data_exception_handling(mock_crypto_handler):
    content = "Sample content to be formatted"

    # Mock the API call to the Gemini model to raise an exception
    with patch('google.generativeai.GenerativeModel') as mock_model_class:
        mock_model = mock_model_class.return_value
        mock_model.generate_content.side_effect = Exception("API Error")
        maker = FineTuningDataMaker()

        with pytest.raises(RuntimeError, match="An error occurred while formatting text: API Error"):
            maker.format_data(content)

# Test large content size handling
def test_large_content_handling(mock_crypto_handler):
    content = "A" * 10001  # Content with more than 10000 tokens
    with patch('wyra.data_maker.tokenizer.encode', return_value=[0] * 10001):  # Mock tokenizer to simulate large token count
        maker = FineTuningDataMaker()
        with pytest.raises(ValueError, match="The text is too large, please split it and make spaced calls."):
            maker.format_data(content)

# Run the tests
if __name__ == "__main__":
    pytest.main()