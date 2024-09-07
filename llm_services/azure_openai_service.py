import os
from openai import AzureOpenAI
from .llm_interface import LLMService
from typing import List, Union
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCall

from utils.image_encoding import encode_image
from utils.retry_logic import retry_request

class AzureOpenAIService(LLMService):
    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)
        
    def initialize_client(self) -> AzureOpenAI:
        """Initialize the Azure OpenAI client with API credentials."""
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            raise EnvironmentError("Azure OpenAI API key or endpoint not found in environment variables.")
        
        api_version = "2023-05-15"

        return AzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=endpoint
        )

    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Make a request to the Azure OpenAI API and handle the response."""
        def api_call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            # Track token usage
            self.token_tracker.update_tokens_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content
        
        return retry_request(api_call)
    
    def make_request_json(self, messages: List[ChatCompletionMessageParam], json_scheme: str) -> str:
        """Make a request to the Azure OpenAI API and return the response in JSON format.
        NOTE: Only use this method if you ONLY expect a JSON response.
        For more details: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode?tabs=python
        and here: https://platform.openai.com/docs/guides/json-mode
        """
        # Check JSON in system message (Required for Azure OpenAI)
        if not ("JSON" or "json" in messages[0].content):
            messages[0].content += " Your task is to return the response in JSON format."
        
        def api_call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={ "type": "json_object" }
            )
            # Track token usage
            self.token_tracker.update_tokens_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content
        
        return retry_request(api_call)
    
    def make_request_with_tools(self, messages: List[ChatCompletionMessageParam], tools: List[ChatCompletionToolParam], parallel_tool_calls: bool = False, return_only_tool_response: bool = False) -> Union[str, List[ChatCompletionMessageToolCall]]:
        """Let the LLM decide which tools to call from a list and return only the tool calls."""
        def api_call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=tools,
                tool_choice="auto", # Let the LLM decide which tool to call
            )
            self.token_tracker.update_tokens_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            
            if return_only_tool_response:
                response_message = response.choices[0].message.tool_calls
            else:
                # Tool calls and content are mutually exclusive, so we can return the content if no tool calls are made
                if not response.choices[0].message.tool_calls:
                    response_message = response.choices[0].message.content
                else:
                    response_message = response.choices[0].message.tool_calls
            return response_message
        
        return retry_request(api_call)
    
    def make_request_image(self, messages: List[ChatCompletionMessageParam], image_path: str) -> str:
        """Give the LLM access to an image file and let it generate a response.
        Doc: https://platform.openai.com/docs/guides/vision
        
        NOTE: OpenAI models will refuse to talk about images where persons are present!
        """
        if not self.model_name in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
            raise ValueError("Use a model with vision capabilities for image requests. The following models support images: gpt-4o, gpt-4o-mini, gpt-4-turbo.")
        
        # Encode the image as base64
        base64_image = encode_image(image_path)
        
        # Modify user prompt to include the image
        user_prompt_content = messages[-1]["content"]
        user_prompt = {"role": "user", "content": [{"type": "text", "text": user_prompt_content}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
        messages[-1] = user_prompt
        
        def api_call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.token_tracker.update_tokens_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content
        
        return retry_request(api_call)