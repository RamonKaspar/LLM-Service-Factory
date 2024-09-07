from llm_services.azure_openai_service import AzureOpenAIService
from llm_services.huggingface_service import HuggingFaceService
from llm_services.openai_service import OpenAIService
from llm_services.llm_interface import LLMService


class LLMServiceFactory():
    """
    Factory class to manage LLM instances.
    This factory ensures that a single instance of an LLM service is created per unique set of parameters (model, provider, temperature, max_tokens).
    """
    _services = {}
    
    @classmethod
    def get_service(cls, model_name: str, provider: str, temperature: float = 1.0, max_tokens: int = 4096) -> LLMService:
        key = (model_name, provider, temperature, max_tokens)
        if key not in cls._services:
            # Check if the model is supported
            if provider == "AzureOpenAI":
                cls._services[key] = AzureOpenAIService(model_name, temperature, max_tokens)
            elif provider == "HuggingFace":
                cls._services[key] = HuggingFaceService(model_name, temperature, max_tokens)
            elif provider == "OpenAI":
                cls._services[key] = OpenAIService(model_name, temperature, max_tokens)
            else:
                # Add support for other providers here, if needed
                raise ValueError(f"Unsupported provider: {provider}. Supported providers: AzureOpenAI, HuggingFace and OpenAI.")
        return cls._services[key]