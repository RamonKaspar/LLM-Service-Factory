import os
import json
from threading import Lock

class TokenTracker:
    _tokens_usage_file = "tokens_usage.json"    # Update this path if you want to save the file in a different location
    _lock = Lock()  # Lock to ensure thread safety
    
    # Cost rates for different models
    # Current values are taken from here: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    # NOTE: Update these values if the cost rates change, or if you add new models
    COST_RATES = {
        "gpt-4o": {"prompt_token_cost": 0.0045 / 1000, "completion_token_cost": 0.0135 / 1000},
        "gpt-35-turbo": {"prompt_token_cost": 0.0018 / 1000, "completion_token_cost": 0.0018 / 1000},
        "gpt-35-turbo-16k": {"prompt_token_cost": 0.0027 / 1000, "completion_token_cost": 0.0036 / 1000}
    }

    @classmethod
    def update_tokens_usage(cls, model_name: str, prompt_tokens: int, completion_tokens: int):
        """Update tokens usage data and save to the JSON file.
        
        Args:
            model_name (str): The name of the LLM model.
            prompt_tokens (int): The number of tokens used for the prompt.
            completion_tokens (int): The number of tokens used for the completion.
        """
        cls._lock.acquire()
        try:
            # Load existing data
            if os.path.exists(cls._tokens_usage_file):
                with open(cls._tokens_usage_file, 'r') as file:
                    tokens_usage = json.load(file)
            else:
                tokens_usage = {}

            # Update the data
            tokens_usage.setdefault(model_name, {"prompt_tokens": 0, "completion_tokens": 0})
            tokens_usage[model_name]["prompt_tokens"] += prompt_tokens
            tokens_usage[model_name]["completion_tokens"] += completion_tokens
            
            # Update overall cost
            if not cls.COST_RATES.get(model_name):
                # We assume no cost for the model if the cost rates are not defined
                cls.COST_RATES[model_name] = {"prompt_token_cost": 0, "completion_token_cost": 0}
            prompt_cost = prompt_tokens * cls.COST_RATES[model_name]["prompt_token_cost"]
            completion_cost = completion_tokens * cls.COST_RATES[model_name]["completion_token_cost"]
            tokens_usage["overall_cost"] = tokens_usage.get("overall_cost", 0) + prompt_cost + completion_cost

            # Save the updated data
            with open(cls._tokens_usage_file, 'w') as file:
                json.dump(tokens_usage, file, indent=4) 
        finally:
            cls._lock.release()  
