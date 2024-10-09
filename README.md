# Wyra

Wyra is a Python library designed to create and format data for fine-tuning OpenAI models, specifically tailored for Azure OpenAI services. It simplifies the process of converting text content into the JSON Lines (JSONL) format required for fine-tuning conversational AI models.

"Wyra" is a term used in indigenous languages to refer to a bird. Just like a "bird", this library will help you "fly" through the implementation of fine-tuning. Enjoy!

## Features

- **Easy usability**: Any text in multiple languages ​​can be used.
- **Easy Integration**: Seamlessly integrates with Azure OpenAI services.
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

## Multiple Texts

Here's an example of how to use Wyra with multiple texts:

```python
# Import necessary libraries
import json
from wyra import FineTuningDataMaker

# Initialize the Fine-Tuning data maker
creator = FineTuningDataMaker()

# Sample contents to format
contents = [
    "First text content here.",
    "Second text content here.",
    "Third text content here."
]

# Create and format data for multiple contents
formatted_data = [creator.format_data(content) for content in contents]

# Save JSONL data to a file
with open('formatted_data.jsonl', 'w') as file:
    for data in formatted_data:
        file.write(data + '\n')
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/sauloleite/wyra/blob/main/LICENSE) file for details.

## Contact

For any questions or feedback, this is my Github [GitHub repositories](https://github.com/sauloleite).

## Other AI Projects

Check out my GitHub Pages: [GitHub IO](https://sauloleite.github.io/).

