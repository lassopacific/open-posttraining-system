import torch
from base_model.qwen import KVCache




device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def generate_text_stream_with_kv_cache(prompt, model, tokenizer, device, max_new_tokens, eos_token_id):
    # Encode prompt and move to device
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    # Set model to evaluation mode
    model.eval()

    # Initialize KV cache
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Initial forward pass using full prompt
    logits = model(input_ids, cache=cache)[:, -1]

    # Autoregressive generation loop
    for _ in range(max_new_tokens):

        # Greedy decoding
        next_token = torch.argmax(logits,dim=-1,keepdim=True)

        # Stop if EOS token is generated
        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break

        # Stream generated token
        yield next_token

        # Next forward pass only uses new token
        logits = model(next_token, cache=cache)[:, -1]


    
         

    
