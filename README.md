# wyra
Wyra is a Python library designed to create and format data for fine-tuning OpenAI models, specifically tailored for Azure OpenAI services. It simplifies the process of converting text content into the JSON Lines (JSONL) format required for fine-tuning conversational AI models.
## Features

- **Easy Integration**: Seamlessly integrates with Azure OpenAI services.
- **Secure Data Handling**: Utilizes encryption for sensitive data.
- **Automated Formatting**: Converts text content into JSONL format effortlessly.
- **Customizable Prompts**: Allows for flexible prompt creation for various use cases.

## Installation

To install Wyra, use pip:

```sh
pip install wyra
```

## Usage

Here's a basic example of how to use Wyra:

```python
from wyra import FineTuningDataMaker

# Initialize the data maker
fine_tunner = FineTuningDataMaker()

# Sample content to format
content = "Your text content here."

# Create and format data
formatted_data = fine_tunner.format_data(content)

print(formatted_data)
# Save JSONL data to a file
with open('formatted_data.jsonl', 'w') as file:
    for item in jsonl_data:
        file.write(f"{item}\n")

```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue on our [GitHub repository](https://github.com/sauloleite/wyra).
