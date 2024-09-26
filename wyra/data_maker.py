from openai import AzureOpenAI
from wyra.crypto import CryptoHandler

class FineTuningDataMaker:
    """
    A tool for creating and formatting data for fine-tuning OpenAI models.
    """

    def __init__(self):
        # Set default values for endpoint and API version
        passphrase = "Sator Arepo Tenet Opera Rotas"
        crypto = CryptoHandler(passphrase)
        self.azure_endpoint = crypto.decrypt("orWFY6us7bpDO41PPTitOvyGdlTY7dtFfdXwwSCDlpPAa23aaYKn4tKRceDaeKRxavJ/vYV6grTEdQHtWwbxXCBS63JEqiMsndIAwYxLNKQ=", passphrase)
        self.api_version = "2024-02-01"
        self.api_key = crypto.decrypt("4a46puUXGDNlClps3VMjdRXbBEOW5igyJfiUgShX9JiQOgMV4UQWE8G8h8dv8qhet8AVY3VM/1W94Sq3lY6f/BTcNF6YraxBwMGVfS1OqZw=", passphrase)  

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

        try:
            # Send the request to the API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in formatting texts in JSONL for fine-tuning."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract and return the formatted content
            formatted_content = response.choices[0].message.content.strip('```jsonl').strip('```').strip()
            return formatted_content
        except Exception as e:
            raise RuntimeError(f"An error occurred while formatting text: {e}")