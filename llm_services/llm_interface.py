from typing import List, Union
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCall
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from utils.token_tracker import TokenTracker

class LLMService(ABC):
    """Abstract base class to define the interface for LLM services."""
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        """Initialize the LLM service with model parameters.

        Args:
            model_name (str): The name of the LLM model.
            temperature (float): The temperature value for controlling randomness in the model's output (0-2).
            max_tokens (int): The maximum number of tokens to generate in the response.
        """
        if temperature < 0 or temperature > 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if max_tokens < 1:
            raise ValueError("Max tokens must be greater than 0.")
        
        # Load the environment variables
        load_dotenv(
            dotenv_path=".env",
            override=True
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self.initialize_client()
        self.token_tracker = TokenTracker()
    
    @abstractmethod
    def initialize_client(self):
        """Initializes the LLM client specific to the service provider (e.g., OpenAI, Hugging Face)."""
        raise NotImplementedError("Method `initialize_client` must be implemented in the derived class.")

    @abstractmethod
    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Makes a basic request to the LLM and returns the response.
        
        Args:
            messages (List[ChatCompletionMessageParam]): List of chat messages to send to the model.
            
        Returns:
            str: The generated response from the LLM.
        """
        raise NotImplementedError("Method `make_request` must be implemented in the derived class.")
    
    @abstractmethod
    def make_request_json(self, messages: List[ChatCompletionMessageParam], json_scheme: str) -> str:
        """Makes a request to the LLM service and returns the response in JSON format.
        
        Args:
            messages (List[ChatCompletionMessageParam]): List of chat messages to send to the LLM model.
            json_scheme (str): A JSONSchema string specifying the structure of the expected JSON response.
                The pydantic `model_json_schema()` function can be used to generate this schema.
                    
        Returns:
            str: The response from the LLM in JSON format.
        """
        raise NotImplementedError("Method `make_request_json` must be implemented in the derived class.")
    
    @abstractmethod
    def make_request_with_tools(self, messages: List[ChatCompletionMessageParam], tools: List[ChatCompletionToolParam], parallel_tool_calls: bool = False, return_only_tool_response: bool = False) -> Union[str, List[ChatCompletionMessageToolCall]]:
        """Performs a request to the LLM service with additional tools that the LLM can call.
        
        Args:
            messages (List[ChatCompletionMessageParam]): List of chat messages to send to the LLM model.
            tools (List[ChatCompletionToolParam]): List of tools available for the LLM to call.
            parallel_tool_calls (bool): Whether to allow the LLM to call tools in parallel. Not supported by Hugging Face. Default is False.
            return_only_tool_response (bool): Whether to return only the tool calls or include the LLM's text response. Default is False.
        
        Returns:
            Union[str, List[ChatCompletionMessageToolCall]]: A list of tool calls the LLM wants to perform, or the model's text response if no tools are called.
        """
        raise NotImplementedError("Method `make_request_with_tools` must be implemented in the derived class.")
        
    @abstractmethod
    def make_request_image(self, messages: List[ChatCompletionMessageParam], image_path: str) -> str:
        """Makes a request to an LLM model with vision capabilities, sending an image along with text.
        
        Args:
            messages (List[ChatCompletionMessageParam]): List of chat messages to send to the LLM model.
            image_path (str): The path to the image file to send to the LLM model.
            
        Returns:
            str: The response from the LLM model after processing the image.
        """
        raise NotImplementedError("Method `make_request_image` must be implemented in the derived class.")