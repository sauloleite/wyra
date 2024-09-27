from openai import AzureOpenAI
from wyra.crypto import CryptoHandler
import json_repair
import json
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

class FineTuningDataMaker:
    """
    A tool for creating and formatting data for fine-tuning OpenAI models.
    """

    def __init__(self):
        # Set default values for endpoint and API version
        crypto = CryptoHandler("wyra")
        self.azure_endpoint = crypto.decrypt("UNrxG56N4oYa4JU72yEoqs/8oGh/euzSATu+t/WBbzthnjMyQNWgZQZItu8RtLNpSe4rCzjInYDqHbjtjtHA0tyaAepd28KFzX5Ooy63jCc=", "wyra")
        self.api_version = "2024-02-01"
        self.api_key = crypto.decrypt("ecOr5rAvJ4vi05qo9h2PzM/8oGh/euzSATu+t/WBbzscKATvs69rxDHX8SjFeJhqxOXp/cJZi8nM2/DcIM4UG8SveFHgD6UbjdXPfWdyGng=", "wyra")  

        # Initialize the AzureOpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

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
            "If you receive inputs in different languages, please return them in the same language."
            "\n\nHere is the content to be formatted:\n\n" + content
        )

        # Calculate the number of tokens in the prompt
        num_tokens = len(tokenizer.encode(prompt))
        if num_tokens > 10000:
            raise ValueError("The text is too large, please split it and make spaced calls.")

        try:
            # Send the request to the API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in formatting texts in JSONL for fine-tuning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            # Extract and return the formatted content
            formatted_content = json_repair.loads(response.choices[0].message.content.strip('```jsonl').strip('```').strip())
            jsonl_content = json.dumps(formatted_content, ensure_ascii=False)
            return jsonl_content
        except Exception as e:
            raise RuntimeError(f"An error occurred while formatting text: {e}")