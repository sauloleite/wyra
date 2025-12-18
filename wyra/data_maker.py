import json
import json_repair
import tiktoken
from google import genai
from google.genai import types

tokenizer = tiktoken.get_encoding("cl100k_base")

class FineTuningDataMaker:
    """
    A tool for creating and formatting data for fine-tuning OpenAI models.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview", client: genai.Client | None = None):
        api_key = "AIzaSyCIqeQuAo4BOD7DaoLmcLbD0z379_-KPJ4"
        if not api_key:
            raise RuntimeError("A Gemini API key must be supplied.")

        self.client = client or genai.Client(api_key=api_key)
        self.model_name = model_name
        
        
    def format_data(self, content):
        """
        Creates and formats data for fine-tuning.

        Parameters:
            content (str): The text content to process.

        Returns:
            str: The formatted JSONL string.
        """
        # Build the prompt to format the content as JSONL
        prompt = (
            "Please format the following as JSON Lines (JSONL) for fine-tuning. Each JSON line should "
            "represent a 'messages' array with the 'role' and 'content' fields, where 'role' is either "
            "'system', 'user', or 'assistant'. Example structure:\n\n"
            '{"messages": [{"role": "system", "content": "<instructions>"}, '
            '{"role": "user", "content": "<user question>"}, '
            '{"role": "assistant", "content": "<assistant response>"}]}'
            "Return only the JSONL-formatted data without any additional text."
            "Please create as many JSONL fields as necessary."
            "If you receive inputs in different languages, please return them in the same language."
            "\n\nHere is the content to be formatted:\n\n" + content
        )

        # Calculate the number of tokens in the prompt
        num_tokens = len(tokenizer.encode(prompt))
        if num_tokens > 10000:
            raise ValueError("The text is too large, please split it and make spaced calls.")

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )
        ]

        try:
            # Issue the generation request with a low temperature for determinism.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.1),
            )

            response_text = None
            if isinstance(getattr(response, "text", None), str) and response.text.strip():
                response_text = response.text
            elif isinstance(getattr(response, "output_text", None), str) and response.output_text.strip():
                response_text = response.output_text

            if not response_text:
                raise RuntimeError("No textual response returned by Gemini API.")

            formatted_content = json_repair.loads(response_text.strip('```jsonl').strip('```').strip())
            jsonl_content = json.dumps(formatted_content, ensure_ascii=False)
            return jsonl_content
        except Exception as e:
            raise RuntimeError(f"An error occurred while formatting text: {e}")