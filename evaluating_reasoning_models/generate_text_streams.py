import torch
from base_model.qwen import KVCache, sample_next_token



device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def generate_text_stream_with_kv_cache(
    prompt,
    model,
    tokenizer,
    device,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.6,
    top_k=20,
    top_p=0.95,
):
    # Encode prompt and move to device
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    # Set model to evaluation mode
    model.eval()

    # Initialize KV cache
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Initial forward pass using full prompt
    logits = model(input_ids, cache=cache)[:, -1]

    stop_token_ids = getattr(tokenizer, "stop_token_ids", None)
    if stop_token_ids is None:
        stop_token_ids = {eos_token_id} if eos_token_id is not None else set()

    # Autoregressive generation loop
    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Stop if a stop token is generated
        if int(next_token.item()) in stop_token_ids:
            break

        # Stream generated token
        yield next_token

        # Next forward pass only uses new token
        logits = model(next_token, cache=cache)[:, -1]
