from enum import Enum

class CostTracker:
    """Track the cost of API calls."""
    @staticmethod
    def track_cost(
        input_tokens_count: int,
        output_tokens_count: int,
        model_name: str,
    ) -> float:
        """Track the cost of an API call."""
        # Define the cost per token for the model
        model_pricing = {
            "gpt-4o": {
                "input": 2.50 / 1_000_000,
                "output": 10.00 / 1_000_000,
            },
            "gpt-4o-2024-08-06": {
                "input": 2.50 / 1_000_000,
                "output": 10.00 / 1_000_000,
            }
        }

        # Get the cost per token for the model
        cost_per_token = model_pricing.get(model_name)
        total_input_cost = input_tokens_count * cost_per_token["input"]
        total_output_cost = output_tokens_count * cost_per_token["output"]
        cost = total_input_cost + total_output_cost
        return cost