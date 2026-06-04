# Generating text with a pre-trained LLM (short)

This folder shows how the project runs local text generation using the `Qwen` model implementation in `base_model/qwen.py` rather than `transformers` high-level helpers.

Key points:
- Artifacts expected: a local folder `generating_text_with_pre_trained_llm/qwen/` containing `tokenizer.json` and `model.safetensors` (the notebook and `generate.py` download these optionally).
- The code uses `Qwen3Tokenizer` (from `base_model.qwen`) and `Qwen3Model` + the local loader `load_hf_weights_into_qwen` to copy tensors from `safetensors` into the model.
- `generate.py` demonstrates the end-to-end flow: load tokenizer, load weights with `safetensors.torch.load_file`, instantiate `Qwen3Model(QWEN_CONFIG_06_B)`, call `load_hf_weights_into_qwen(...)`, move the model to device and `torch.bfloat16`, and call `generate_text(...)` to produce output. The script expects the `qwen/` folder next to `generate.py`.

What the notebook (`main.ipynb`) shows:
- Simple streaming generation loops that call the model step-by-step (argmax sampling) or use the project's `generate_text(...)` helper.
- A KV-cache-based generator that pre-fills the prompt and then decodes one token at a time using cached K/V tensors for faster autoregressive decoding.
- Optional use of `torch.compile(model)` to benchmark compiled inference and reporting via `generate_stats`.

Quick usage (run from `generating_text_with_pre_trained_llm`):

```bash
# ensure artifacts are in ./qwen/ (tokenizer.json, model.safetensors, config.json)
python generate.py   # script prints the generated response (default prompt in the script)
```

Or follow the notebook `main.ipynb` for interactive examples (streaming, KV cache, compilation, and basic timing).

If you want, I can also simplify `main.ipynb` examples or update the notebook so the cells call `generate.py` functions directly.  
- Simple parameter to tune
- Effective diversity control
- Well-understood behavior

**Cons:**
- May generate nonsense at high temps
- Temperature is somewhat arbitrary

**Best for:** General text generation, creative tasks

### 3. Top-k Sampling
Select from the k most likely next tokens.

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    top_k=50  # Consider top 50 tokens
)
```

**How it works:**
1. Sort tokens by probability
2. Keep only top k tokens
3. Renormalize probabilities
4. Sample from this reduced set

**Pros:**
- Prevents very unlikely tokens
- Maintains diversity
- Consistent with all temperatures

**Cons:**
- Fixed k for all contexts
- May be too restrictive or loose

**Best for:** Balanced quality and diversity

### 4. Top-p (Nucleus Sampling)
Select from the smallest set of tokens with cumulative probability ≥ p.

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9  # Accumulate to 90% probability
)
```

**How it works:**
1. Sort tokens by probability
2. Add tokens to set until cumulative probability ≥ p
3. Sample from this adaptive set

**Typical value:** p=0.9 (include tokens making up 90% of probability mass)

**Pros:**
- Adapts to model confidence
- High quality + diversity balance
- Prevents low-probability tokens

**Cons:**
- More complex than top-k
- Still requires tuning p value

**Best for:** High-quality, diverse generation

### 5. Combined Strategies
Use multiple techniques together for best results.

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)
```

**Strategy combinations:**
- Temperature + Top-p: Very common, excellent quality
- Temperature + Top-k: Good alternative
- All three together: Maximum control

## Generation Parameters

### Common Parameters

#### `max_new_tokens` (Required)
Maximum number of new tokens to generate.

```python
# Generate up to 100 new tokens
max_new_tokens=100
```

Typical values:
- 50-100: Short answers
- 200-500: Medium responses
- 1000+: Long form generation

#### `temperature`
Controls randomness (0.1-2.0, default 1.0).

```python
temperature=0.7  # Lower = more focused
```

#### `top_k`
Consider only top k tokens (default None, no limit).

```python
top_k=50  # Consider 50 most likely tokens
```

#### `top_p`
Nucleus sampling threshold (default 1.0).

```python
top_p=0.9  # Include tokens up to 90% probability
```

#### `repetition_penalty`
Penalize repeated tokens (default 1.0, no penalty).

```python
repetition_penalty=1.2  # Discourage repetition
```

#### `do_sample`
Use sampling vs. greedy (default False).

```python
do_sample=True  # Enable sampling
do_sample=False  # Greedy decoding
```

#### `num_return_sequences`
Generate multiple sequences per prompt.

```python
num_return_sequences=3  # Generate 3 options per prompt
```

## Advanced Usage

### Batch Generation

```python
# Generate for multiple prompts at once
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,
    batch_size=3
)

results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Chain-of-Thought Generation

```python
# Prompt model to show reasoning steps
cot_prompt = """
Q: Solve the following problem step by step.
Problem: {}

Solution:
Let me think through this step by step:
1.
2.
3.

Final Answer:
"""

prompt = cot_prompt.format("2 + 2")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs["input_ids"], max_new_tokens=200)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Multiple Generation Runs

```python
# Generate multiple samples and take best/majority vote
def generate_multiple(prompt, num_samples=5, temperature=0.9):
    results = []
    for _ in range(num_samples):
        outputs = model.generate(
            tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda"),
            max_new_tokens=100,
            do_sample=True,
            temperature=temperature
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result)
    return results

# Get 5 different generations
samples = generate_multiple("Solve: 2+2=", num_samples=5)
```

## Performance Optimization

### 1. Use Smaller Models for Testing
```python
# Fast testing
model_name = "Qwen/Qwen-1.8B"

# Production/quality
model_name = "Qwen/Qwen-7B"
```

### 2. Enable Flash Attention (if available)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)
```

### 3. Batch Multiple Prompts
```python
# Efficient batch processing
prompts = ["..."] * 100  # 100 prompts
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
```

### 4. Use Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
```

### 5. Reduce Precision (4-bit/8-bit)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
```

## Collecting Statistics

### Basic Statistics
```python
import time
import torch

def generate_with_stats(model, tokenizer, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()
    
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    gen_time = time.time() - start_time
    memory_used = torch.cuda.memory_allocated() - start_memory
    
    text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
    
    return {
        "text": text,
        "tokens": len(outputs["sequences"][0]),
        "time": gen_time,
        "memory_mb": memory_used / (1024**2),
        "tokens_per_second": len(outputs["sequences"][0]) / gen_time
    }

stats = generate_with_stats(model, tokenizer, "What is AI?")
```

### Use `generate_stats.py`
```bash
python generate_stats.py \
    --results "generation_results.json" \
    --output "statistics.json" \
    --compute_throughput true
```

## Common Use Cases

### Use Case 1: Quick Testing
```bash
python generate.py --prompt "Hello, world" --model "Qwen/Qwen-1.8B" --max_tokens 50
```

### Use Case 2: Generate Predictions for Evaluation
```bash
python generate.py \
    --input_file "problems.json" \
    --output_file "predictions.jsonl" \
    --batch_size 32
```

### Use Case 3: Compare Decoding Strategies
```bash
for strategy in "greedy" "top_p" "top_k" "temperature"; do
    python generate.py --strategy "$strategy" --output "results_$strategy.json"
done
```

### Use Case 4: Collect Performance Metrics
```bash
python generate.py --collect_stats true --output_stats "stats.json"
```

## Troubleshooting

### Issue: OOM (Out of Memory) Error
**Solution:**
- Reduce `max_new_tokens`
- Reduce batch size
- Use smaller model variant
- Enable quantization

### Issue: Generation is Slow
**Solution:**
- Use smaller model
- Reduce `max_new_tokens`
- Batch multiple prompts
- Enable optimization (flash attention, etc.)

### Issue: Generations are Repetitive
**Solution:**
- Increase `temperature`
- Use `repetition_penalty > 1.0`
- Reduce `top_k` value
- Enable nucleus sampling (top_p)

### Issue: Generations are Incoherent
**Solution:**
- Decrease `temperature`
- Increase `top_k` or decrease `top_p`
- Use smaller models might be more stable
- Improve prompt engineering

## Next Steps

After generating predictions:

1. **Evaluate quality** → Go to `../evaluating_reasoning_models/`
   - Compare predictions with ground truth
   - Measure accuracy on benchmarks

2. **Improve reasoning** → Go to `../improving_reasoning_with_inference_time_scaling/`
   - Apply inference optimizations
   - Use self-consistency, CoT, etc.

3. **Fine-tune model** → Create training pipeline
   - Use generated outputs as baselines
   - Train on specific domains
   - Re-evaluate with this script

## Reference

- [Hugging Face Generation Documentation](https://huggingface.co/docs/transformers/generation_strategies)
- [Temperature and Sampling](https://huggingface.co/blog/how-to-generate)
- [Nucleus Sampling Paper](https://arxiv.org/abs/1904.09751)
- [Beam Search & Alternatives](https://arxiv.org/abs/1904.09751)

## Questions?

- Check script docstrings and help text
- Refer to main README for project overview
- See examples in Jupyter notebooks
