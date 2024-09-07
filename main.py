from LLMServiceFactory import LLMServiceFactory
from llm_services.llm_interface import LLMService


def main():
    """
    Demonstrates how to use the LLMServiceFactory to create a language model client 
    and interact with the LLM using basic functionalities.
    
    We'll be using the OpenAI "gpt-4o-mini" model for demonstration. 
    You can easily switch between services (Azure, Hugging Face) by changing the provider.
    """
    
    # Instantiate an OpenAI client using the LLMServiceFactory
    client = LLMServiceFactory.get_service("gpt-4o-mini", "OpenAI")

    # For Azure OpenAI, change the provider to "AzureOpenAI":
    # client = LLMServiceFactory.get_service("gpt-4o", "AzureOpenAI")

    # For Hugging Face, change the provider to "HuggingFace":
    # client = LLMServiceFactory.get_service("meta-llama/Meta-Llama-3-8B-Instruct", "HuggingFace")

    
    # 1. Perform a simple request
    simple_request(client)

    # 2. Perform a request expecting a JSON response
    request_with_json(client)

    # 3. Perform a request using tools
    request_with_tools(client)

    # 4. (Optional) If your model supports vision, send an image
    request_with_image(client, "path/to/your/image.jpg")


def simple_request(client: LLMService):
    """
    Perform a basic text completion request.
    """
    print("\n--- Simple Request ---")
    question = "What is the capital of France?"
    response = client.make_request([{"role": "user", "content": question}])
    print(f"Question: {question}")
    print(f"Response: {response}\n")


def request_with_json(client: LLMService):
    """
    Request expecting a JSON response.
    """
    print("\n--- Request with JSON ---")
    schema = {
        "type": "object",
        "properties": {
            "response": {"type": "string"}
        },
        "required": ["response"]
    }
    content = "What is the capital of France? Answer in JSON format: {\"response\": \"Your answer\"}"
    response = client.make_request_json([{"role": "user", "content": content}], schema)
    print(f"Response (JSON): {response}\n")


def request_with_tools(client: LLMService):
    """
    Demonstrates how to use tools (function calling) in an LLM request.
    """
    print("\n--- Request with Tools ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_delivery_date",
                "description": "Retrieve delivery date for a customer order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Customer's order ID."}
                    },
                    "required": ["order_id"],
                },
            }
        }
    ]
    content = "When will my package arrive? My order ID is 12345."
    response = client.make_request_with_tools([{"role": "user", "content": content}], tools)
    print(f"Response with Tools: {response}\n")


def request_with_image(client: LLMService, image_path: str):
    """
    Demonstrates how to send an image to the LLM for vision tasks.
    Only for models with vision capabilities (e.g., OpenAI GPT-4 with vision).
    """
    print("\n--- Request with Image ---")
    try:
        response = client.make_request_image([{"role": "user", "content": "Describe the image."}], image_path)
        print(f"Image Response: {response}\n")
    except NotImplementedError:
        print(f"{client.__class__.__name__} does not support image requests.\n")


if __name__ == "__main__":
    main()
