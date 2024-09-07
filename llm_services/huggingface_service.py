import os
from huggingface_hub import InferenceClient
from .llm_interface import LLMService
from typing import List, Union
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCall

from utils.retry_logic import retry_request

class HuggingFaceService(LLMService):
    """
    Inference models from the free Hugging Face serverless API. Doc: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client
    
    The models `mistralai/Mistral-7B-Instruct-v0.2`, `HuggingFaceH4/zephyr-7b-beta`, `meta-llama/Meta-Llama-3-8B-Instruct`, 
    `codellama/CodeLlama-34b-Instruct-hf`, `microsoft/Phi-3-mini-4k-instruct` and `mistralai/Mixtral-8x7B-Instruct-v0.1` perform 
    well for instructive text generation.
    """
    
    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)
        
    def initialize_client(self):
        # Check if the model is available
        if self.model_name not in self._get_available_models():
            raise ValueError(f"Model '{self.model_name}' is not available. Call `_get_available_models` to see all available models.")
        
        return InferenceClient(self.model_name, token=os.getenv("HUGGINGFACE_API_TOKEN"))
    
    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Make a request to the Hugging Face API and handle the response."""
        def api_call():
            response = self.client.chat_completion(
                messages=messages, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens
            )
            # Track token usage
            self.token_tracker.update_tokens_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content

        return retry_request(api_call)
    
    def make_request_json(self, messages: List[ChatCompletionMessageParam], json_scheme: str) -> str:
        """Make a request to the Hugging Face API and return the response in JSON format.
        NOTE: Only use this method if you ONLY expect a JSON response.
        For more details: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.generate_json
        and here: https://huggingface.co/docs/text-generation-inference/en/guidance"""
        def api_call():
            response = self.client.chat_completion(
                messages=messages, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
                response_format={"type": "json", "value": json_scheme},
            )
            # Track token usage
            self.token_tracker.update_tokens_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content

        return retry_request(api_call)
    
    def make_request_with_tools(self, messages: List[ChatCompletionMessageParam], tools: List[ChatCompletionToolParam], parallel_tool_calls: bool = False, return_only_tool_response: bool = False) -> Union[str, List[ChatCompletionMessageToolCall]]:
        """Let the LLM decide which tools to call from a list and return only the tool calls.
        NOTE: Hugging Face models do not support parallel tool calls!""" 
        if parallel_tool_calls:
            raise ValueError("Hugging Face models do not support parallel tool calls.")
               
        # The Huggingface client crashes if `enum` is an empty list, so we have to remove them
        for tool in tools:
            parameters = tool['function']['parameters']
            for param_name, param in parameters['properties'].items():
                if 'enum' in param and not param['enum']:
                    del param['enum']
                    
        def api_call():
            response = self.client.chat_completion(
                messages=messages, 
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
                    # Tool calls and content are mutually exclusive, so we can return the content if no tool calls are made
                    if not response.choices[0].message.tool_calls:
                        response_message = response.choices[0].message.content
                    else:
                        response_message = response.choices[0].message.tool_calls
            return response_message

        return retry_request(api_call)
                
    def make_request_image(self, messages: List[ChatCompletionMessageParam], image_path: str) -> str:
        """Give the LLM access to an image file and let it generate a response."""
        raise NotImplementedError("Method `make_request_image` is not implemented for HuggingFace models.")
    
    def _get_available_models(self):
        """Get a list of all available models from the Hugging Face API.
        See here: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.list_deployed_models
        """
        client = InferenceClient(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        models = client.list_deployed_models()
        return models["text-generation"]    # We are only interested in text generation models