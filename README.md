# LLM Service Factory

LLM Service Factory is a Python module designed to simplify the interaction with multiple large language model (LLM) service providers. It supports Azure OpenAI, OpenAI, and Hugging Face, with built-in functionality for automatic token usage tracking. The architecture is modular and extendable, allowing easy addition of new providers, models, and functionality in the future.

## Features

- **Multi-provider Support:** Azure OpenAI, Hugging Face, and OpenAI services are currently supported.
- **Singleton Design:** Ensures only one instance of an LLM service is created per unique configuration.
- **Token Usage Tracking:** Automatically tracks the number of tokens used by each model and calculates the cost, stored in a `tokens_usage.json` file.
- **Extensible:** Easily add new providers and models by extending the base class.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/llm-service-factory.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables: Create a `.env` file in the project root directory to store your API keys and other configurations. You can use the `.env.example` file as a template.

## Usage

### Initializing Services

You can use the `LLMServiceFactory` to get an instance of the desired LLM service. Each service instance is a singleton, meaning that the same instance is reused if requested with the same parameters.

```python
from LLMServiceFactory import LLMServiceFactory

# Initialize an Azure OpenAI service
azure_service = LLMServiceFactory.get_service(
    model_name="gpt-4o",
    provider="AzureOpenAI"
)

# Initialize a Hugging Face service
hf_service = LLMServiceFactory.get_service(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    provider="HuggingFace"
)

# You can also use the OpenAI provider or easily add a new one by extending the llm_interface.py base class
```

### Making Requests

Once you have the service instance, you can use it to make requests to the LLMs. Each service supports different request types like chat completions, JSON completions, tool usage, and even vision-related tasks.

```python
# Make a request to Azure OpenAI
messages = [{"role": "user", "content": "Explain the theory of relativity."}]
response = azure_service.make_request(messages)
print(response)
```

### Token Usage Tracking

Token usage is automatically tracked for each model and stored in tokens_usage.json. The tracking includes both prompt and completion tokens, and the total cost based on predefined rates.

```json
{
  "gpt-4o": {
    "prompt_tokens": 410020,
    "completion_tokens": 64425
  },
  "overall_cost": 2.978635499999999,
  "gpt-35-turbo-16k": {
    "prompt_tokens": 90032,
    "completion_tokens": 5756
  }
}
```

### Adding a New Provider

To add support for a new LLM provider (e.g., Google Gemini):

1. Create a class that extends the `llm_service/llm_interface.py` base class.
2. Implement the `initialize_client`, `make_request`, and other required methods.
3. Register the new service in the LLMServiceFactory.
4. Add the necessary API keys in the `.env` file.
5. Add cost rates in the `utils/token_tracker.py` file (if you want to track the costs).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
