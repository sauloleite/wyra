import json
from unittest.mock import MagicMock, patch

import pytest

from wyra.data_maker import FineTuningDataMaker


def test_initialization():
    with patch("wyra.data_maker.genai.Client") as mock_client_cls:
        maker = FineTuningDataMaker()
        mock_client_cls.assert_called_once_with(api_key="AIzaSyCIqeQuAo4BOD7DaoLmcLbD0z379_-KPJ4")
        assert maker.model_name == "gemini-3-flash-preview"


def test_format_data():
    content = "Sample content to be formatted"
    formatted_response = '{"messages": [{"role": "system", "content": "<instructions>"}]}'

    client_mock = MagicMock()
    client_mock.models.generate_content.return_value = MagicMock(
        text=f"```jsonl\n{formatted_response}\n```"
    )

    maker = FineTuningDataMaker(client=client_mock)
    result = maker.format_data(content)

    prompt_part = "Please format the following as JSON Lines (JSONL) for fine-tuning."
    called_contents = client_mock.models.generate_content.call_args.kwargs["contents"]

    assert prompt_part in called_contents[0].parts[0].text
    assert json.loads(result) == json.loads(formatted_response)


def test_format_data_exception_handling():
    client_mock = MagicMock()
    client_mock.models.generate_content.side_effect = Exception("API Error")

    maker = FineTuningDataMaker(client=client_mock)

    with pytest.raises(RuntimeError, match="An error occurred while formatting text: API Error"):
        maker.format_data("Sample content")


def test_large_content_handling():
    content = "A" * 10001
    with patch("wyra.data_maker.tokenizer.encode", return_value=[0] * 10001):
        maker = FineTuningDataMaker(client=MagicMock())
        with pytest.raises(ValueError, match="The text is too large, please split it and make spaced calls."):
            maker.format_data(content)


def test_no_text_response():
    client_mock = MagicMock()
    client_mock.models.generate_content.return_value = MagicMock(text="")
    maker = FineTuningDataMaker(client=client_mock)

    with pytest.raises(RuntimeError, match="No textual response returned by Gemini API."):
        maker.format_data("Sample content")


if __name__ == "__main__":
    pytest.main()