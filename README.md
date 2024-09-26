# Wyra

Wyra is a Python library designed to create and format data for fine-tuning OpenAI models, specifically tailored for Azure OpenAI services. It simplifies the process of converting text content into the JSON Lines (JSONL) format required for fine-tuning conversational AI models.

"Wyra" is a term used in indigenous languages to refer to a bird or bird species. Just like a bird, this library will help you "fly" through fine-tuning. Enjoy!

## Features

- **Easy Integration**: Seamlessly integrates with Azure OpenAI services.
- **Easy usability**: Any text in multiple languages ​​can be used.
- **Secure Data Handling**: Utilizes encryption for sensitive data.
- **Automated Formatting**: Converts text content into JSONL format effortlessly.
- **Customizable Prompts**: Allows for the creation of flexible fine-tuning datasets for various use cases.

## Installation

To install Wyra, use pip:

```sh
pip install wyra
```

## Usage

Here's a basic example of how to use Wyra:

```python
# Import necessary libraries
import json
from wyra import FineTuningDataMaker

# Initialize the Fine-Tuning data maker
creator = FineTuningDataMaker()

# Sample content to format
content = "Your text content here (You can use any text in any language you want)."

# Create and format data
formatted_data = creator.format_data(content)

# Save JSONL data to a file
with open('formatted_data.jsonl', 'w') as file:
    file.write(formatted_data)
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/sauloleite/wyra/blob/main/LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue on our [GitHub repository](https://github.com/sauloleite/wyra).

