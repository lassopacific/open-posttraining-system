# Evaluating Reasoning Models

## Overview

This folder contains a comprehensive evaluation framework for benchmarking and assessing the reasoning capabilities of language models. Use this to:

- **Measure reasoning accuracy** - How well does the model solve math/logic problems?
- **Test different strategies** - Compare greedy decoding vs. self-consistency
- **Generate evaluation reports** - Detailed metrics and analysis
- **Compare model versions** - Track improvement across different models or fine-tuning runs

**Primary benchmark:** MATH500 dataset (500 mathematical reasoning problems)

## Contents

### Core Scripts

#### `evaluating_reasoning_models.py`
Main evaluation pipeline that runs your model on benchmark datasets and collects results.

**Key responsibilities:**
- Load model and tokenizer
- Load benchmark dataset (MATH500)
- Generate predictions using the model
- Compare predictions with ground truth
- Calculate accuracy metrics
- Generate evaluation reports

**Usage:**
```bash
python evaluating_reasoning_models.py --model "path/to/model" --output "results.jsonl"
```

#### `model_and_tokenizer.py`
Utilities for loading pre-trained models and tokenizers.

**Key functions:**
- `load_model()` - Load model from Hugging Face or local path
- `load_tokenizer()` - Load corresponding tokenizer
- Handle GPU/CPU device selection
- Manage model configuration

**Usage:**
```python
from model_and_tokenizer import load_model, load_tokenizer
model = load_model("Qwen/Qwen-7B")
tokenizer = load_tokenizer("Qwen/Qwen-7B")
```

#### `text_generation_wrapper.py`
Wrapper for generating text from models with various decoding strategies.

**Key functions:**
- `generate_greedy()` - Deterministic generation
- `generate_with_sampling()` - Temperature/top-p sampling
- `generate_chain_of_thought()` - CoT prompting
- Batch generation for efficiency

**Supports:**
- Temperature scaling
- Top-k and top-p sampling
- Maximum token limits
- Early stopping conditions

#### `load_math_500.py`
Utilities for loading and processing the MATH500 benchmark dataset.

**Key functions:**
- `load_dataset()` - Load all 500 problems
- `load_split()` - Load specific train/test splits
- `format_problem()` - Format problem for model input
- `extract_answer()` - Extract final answer from problem text

### Datasets

#### `math500_test.json`
Test set of 500 mathematical problems in JSON format.

**Structure:**
```json
{
  "problem_id": "1",
  "problem": "Find the value of...",
  "solution": "Step-by-step solution...",
  "answer": "42"
}
```

**Properties:**
- 500 diverse math problems
- Difficulty range: algebra to competition mathematics
- Ground truth answers included
- Solutions with explanations

#### `math500-base-cuda-cot-greedy.jsonl`
Evaluation results: Model predictions using Chain-of-Thought (CoT) with greedy decoding.

**Format:** One JSON object per line (JSONL)
```json
{"problem_id": "1", "prediction": "42", "gold_answer": "42", "correct": true, "reasoning": "..."}
```

**Key metrics in results:**
- `correct` - Whether prediction matches ground truth
- `prediction` - Model's final answer
- `reasoning` - Model's reasoning process (if CoT)

#### `math500-base-cuda-cot-self-consistency.jsonl`
Evaluation results: Model predictions using Chain-of-Thought + Self-Consistency (multiple reasoning paths).

**Comparison:**
- **Greedy:** Single reasoning path, deterministic
- **Self-Consistency:** Multiple reasoning paths, take majority vote
- Usually achieves higher accuracy due to diversity

## Workflow

### Step 1: Prepare Model

Ensure you have:
1. Downloaded model weights (via `../downloading_the_base_model/`)
2. Custom architecture initialized (via `../base_model/`)

```python
# Load model and tokenizer
from model_and_tokenizer import load_model, load_tokenizer
model = load_model("Qwen/Qwen-7B")
tokenizer = load_tokenizer("Qwen/Qwen-7B")
```

### Step 2: Load Benchmark

```python
from load_math_500 import load_dataset
problems = load_dataset("math500_test.json")
print(f"Loaded {len(problems)} problems")
```

### Step 3: Generate Predictions

```python
from text_generation_wrapper import generate_chain_of_thought
predictions = []
for problem in problems:
    output = generate_chain_of_thought(
        model=model,
        tokenizer=tokenizer,
        prompt=problem["problem"],
        max_tokens=500
    )
    predictions.append(output)
```

### Step 4: Evaluate Results

```python
from evaluating_reasoning_models import evaluate
results = evaluate(predictions, problems)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Correct: {results['correct']} / {results['total']}")
```

### Step 5: Analyze and Compare

Compare different strategies:
- **Greedy decoding** - Fast, deterministic
- **Sampling + CoT** - Better reasoning, slower
- **Self-consistency** - Highest accuracy, most expensive

## Key Metrics

### Primary Metric: Accuracy
- **Definition:** Percentage of predictions matching ground truth
- **Formula:** Correct predictions / Total problems
- **Example:** 350/500 correct = 70% accuracy

### Supporting Metrics
- **Pass rate** - Percentage of correctly solved problems
- **Average reasoning length** - Tokens in model's explanation
- **Time per problem** - Latency for inference
- **Total compute** - GPU hours for evaluation

## Evaluation Strategies

### 1. Greedy Decoding (Baseline)
```python
# Deterministic, always picks most likely token
# Fast but limited exploration
accuracy = evaluate_greedy(model, problems)
```

**Pros:** Fast, reproducible, simple
**Cons:** Limited diversity, may miss correct answers

### 2. Chain-of-Thought (CoT)
```python
# Encourage step-by-step reasoning before final answer
# Usually improves accuracy on reasoning tasks
accuracy = evaluate_cot(model, problems)
```

**Pros:** Better for complex reasoning, interpretable
**Cons:** Slower, may introduce errors in reasoning

### 3. Self-Consistency
```python
# Generate K reasoning paths, take majority vote
# Very effective for math/logic problems
accuracy = evaluate_self_consistency(model, problems, num_samples=5)
```

**Pros:** Highest accuracy often achieved
**Cons:** K times slower than greedy (K=5-10x cost)

### 4. Temperature Scaling
```python
# Adjust randomness of predictions
# Low temp = deterministic, High temp = diverse
accuracy = evaluate_with_temperature(model, problems, temperature=0.7)
```

**Typical values:**
- 0.1-0.3: Focused, deterministic
- 0.7-1.0: Balanced
- 1.5-2.0: Very creative/diverse

## Running Evaluation

### Full Evaluation Pipeline

```bash
python evaluating_reasoning_models.py \
    --model "Qwen/Qwen-7B" \
    --dataset "math500_test.json" \
    --strategy "cot-self-consistency" \
    --num_samples 5 \
    --output "results.jsonl"
```

### Evaluate Specific Strategy

```bash
# Greedy
python evaluating_reasoning_models.py --strategy greedy

# CoT
python evaluating_reasoning_models.py --strategy cot

# Self-Consistency (K=5)
python evaluating_reasoning_models.py --strategy self-consistency --k 5
```

### Evaluate Multiple Models

```bash
# Compare different model sizes
for model in "Qwen/Qwen-1.8B" "Qwen/Qwen-7B" "Qwen/Qwen-14B"; do
    python evaluating_reasoning_models.py --model "$model"
done
```

## Interpreting Results

### Understanding JSONL Results

Each line contains one prediction:
```json
{
    "problem_id": "42",
    "problem": "Solve: 2x + 5 = 13",
    "ground_truth": "x = 4",
    "prediction": "x = 4",
    "correct": true,
    "confidence": 0.95,
    "reasoning": "2x + 5 = 13\nSubtract 5: 2x = 8\nDivide by 2: x = 4"
}
```

### Common Patterns
- **Low accuracy:** Model underfitting or task too hard
- **High accuracy:** Model well-suited for domain
- **Inconsistent accuracy:** Check for data distribution shifts
- **Long reasoning chains:** Model overexplaining (check max_tokens)

## Comparison Examples

### Example 1: Model Size Comparison
```
Model          | Greedy | CoT   | Self-Consistency
Qwen-1.8B      | 42%    | 48%   | 52%
Qwen-7B        | 58%    | 68%   | 75%
Qwen-14B       | 72%    | 82%   | 87%
```

### Example 2: Strategy Comparison (Same Model)
```
Strategy              | Accuracy | Time/Problem | Cost
Greedy (baseline)     | 58%      | 0.5s         | 1x
CoT                   | 68%      | 2.0s         | 4x
Self-Consistency (K=5)| 75%      | 10.0s        | 20x
```

## Benchmarking Best Practices

1. **Use consistent hardware** - Same GPU for fair comparison
2. **Run multiple times** - Average results to reduce variance
3. **Record all parameters** - Temperature, max_tokens, sampling strategy
4. **Save detailed results** - Keep JSONL for analysis
5. **Track improvements** - Compare before/after fine-tuning

## Troubleshooting

### Issue: Out of Memory During Evaluation
**Solution:**
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model variant
- Evaluate fewer problems first

### Issue: Predictions are always the same
**Problem:** Model might be stuck or copying prompt
**Solution:**
- Check model loading
- Increase temperature
- Verify tokenizer working correctly

### Issue: Very low accuracy
**Problem:** Model might not understand task format
**Solution:**
- Check problem formatting
- Try different prompt templates
- Verify model downloaded correctly
- Use better-suited model

## Next Steps

After evaluating baseline performance:

1. **Improve reasoning** → Go to `../improving_reasoning_with_inference_time_scaling/`
   - Apply inference-time techniques (CoT, self-consistency)
   - Measure accuracy improvements

2. **Fine-tune model** → Create training pipeline
   - Use MATH500 for training data
   - Fine-tune on reasoning tasks
   - Re-evaluate with this pipeline

3. **Analyze errors** → Study failure cases
   - Which problems are hardest?
   - Common error patterns?
   - Improve prompting or model?

## Reference & Resources

- [MATH Dataset Paper](https://arxiv.org/abs/2103.03874)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Temperature Scaling in LLMs](https://arxiv.org/abs/2102.07033)

## Questions?

- Refer to script docstrings for detailed function signatures
- Check the main README for project overview
- Look at result JSONL files for output format examples
