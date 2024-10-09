from wyra.crypto import CryptoHandler
import json_repair
import json
import tiktoken
import google.generativeai as genai  

tokenizer = tiktoken.get_encoding("cl100k_base")

class FineTuningDataMaker:
    """
    A tool for creating and formatting data for fine-tuning OpenAI models.
    """

    def __init__(self):
        # Set default values for API key
        crypto = CryptoHandler("wyra")
        self.api_key = crypto.decrypt("tZOqzT68nu6g7INzcmNMtWZRfIMBMhMTJj3DAn66uUcfLYv/Ftcz2SW+uS6F4zmKSQBN/vBBknBJRh6LnScNabgYqq6YY/vGXsJGud82kyY=", "wyra")

        # Configure the Gemini API with your key
        genai.configure(api_key=self.api_key)

        # Create the Gemini model (equivalente ao GPT da OpenAI)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        
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

        try:
            # Send the request to the API
            response = self.model.generate_content(prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                ),
            )
            print(response.text)
            # Extract and return the formatted content
            formatted_content = json_repair.loads(response.text.strip('```jsonl').strip('```').strip())
            jsonl_content = json.dumps(formatted_content, ensure_ascii=False)
            return jsonl_content
        except Exception as e:
            raise RuntimeError(f"An error occurred while formatting text: {e}")