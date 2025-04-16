from transformers import LogitsProcessor
import torch

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Keep only allowed token scores, mask the rest
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask

def get_allowed_tokens(tokenizer):
    """
    Returns a list of allowed tokens like 'Document0', 'Document1', ..., 'Document9'
    and corresponding token IDs for controlling output generation.
    """
    token_strings = [f"Document{i}" for i in range(10)] + [","]  # Adjust range if needed
    token_ids = tokenizer.convert_tokens_to_ids(token_strings)
    return token_ids

