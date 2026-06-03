# Open Post Training

An open-source research engineering project focused on the post-training stack for large language models: supervised fine-tuning, preference optimization, reinforcement learning, reasoning behaviors, evaluation, and scalable inference systems.

The goal of this repository is to build research-grade, reproducible implementations of modern post-training pipelines used in frontier language models, while emphasizing clarity, experimentation, and systems understanding.

---

## 🎯 Current Focus: Improving Reasoning with Inference Time Scaling

**Status:** Active Development

This project is currently focused on **inference-time scaling techniques** to improve reasoning capabilities of language models. Instead of just scaling parameters at training time, we explore how model behavior can be enhanced during inference through sophisticated decoding strategies and sampling techniques.

### Key Techniques Under Investigation

#### **1. Temperature Scaling**
Temperature controls the randomness of model predictions. 
- **Low temperature (0.1-0.3):** More deterministic, focused responses. Useful for factual questions where consistency matters.
- **High temperature (0.8-2.0):** More diverse, creative responses. Better for exploration and generating varied reasoning paths.
- **Formula:** `logits_scaled = logits / temperature` — higher temperature flattens the probability distribution, lower temperature sharpens it.

#### **2. Top-p (Nucleus Sampling)**
Instead of selecting from the top-k most likely tokens, top-p selects from the smallest set of tokens whose cumulative probability exceeds a threshold p (typically 0.9).
- Dynamically adjusts the vocabulary size based on the confidence of predictions
- Prevents selecting low-probability outliers while maintaining diversity
- Example: If top-1 token has 60% probability and top-2 has 35%, with p=0.9, both are included; low-probability tokens are filtered out

#### **3. Self-Consistency (Majority Voting)**
Generate multiple reasoning chains independently and take the majority vote of the final answers.
- Improves accuracy on reasoning tasks by averaging out errors in individual chains
- Particularly effective for mathematical and logical reasoning
- Key insight: Different reasoning paths can arrive at the same correct answer

#### **4. Chain-of-Thought (CoT) Prompting**
Explicitly encourage step-by-step reasoning in model outputs before arriving at the final answer.
- Models perform better on complex reasoning when forced to "show their work"
- Compatible with all other scaling techniques

#### **5. Token Probability Thresholding**
Only accept tokens that exceed a minimum confidence threshold, allowing for early stopping or fallback strategies.
- Useful for real-time applications where quality must be guaranteed
- Can reject low-confidence outputs and trigger resampling

### Why Inference Time Scaling Matters
- **No retraining required:** Apply techniques to existing models immediately
- **Cost-effective:** Trade compute at inference time for better reasoning
- **Flexible:** Adapt behavior per query without model modification
- **Complementary:** Works alongside model fine-tuning for cumulative improvements

---

## 📊 Key Experimental Findings

### Benchmark Results (MATH-500)

We evaluated multiple inference-time strategies on the MATH-500 mathematical reasoning dataset:

#### Setup: First 20 Problems from MATH-500
- **Hardware:** Tesla T4 GPU
- **Model:** Qwen-7B (base and reasoning variants)
- **Prompt:** Chain-of-Thought with mathematical problem template
- **Metrics:** Accuracy, runtime, computational cost

#### Results

| Model | Strategy | Samples | Accuracy | Runtime | Cost vs Baseline |
|-------|----------|---------|----------|---------|------------------|
| **Base** | Greedy CoT | 1 | **15.0%** (3/20) | ~5.8 min | 1x (baseline) |
| **Base** | CoT + Self-Consistency (K=7) | 7 | **20.0%** (4/20) | ~36.6 min | **6.3x** |
| **Reasoning** | Greedy CoT | 1 | **15.0%** (3/20) | ~5.8 min | 1x |

### Critical Insights

**1. Self-Consistency Shows Limited Gains**
- Self-consistency voting improved accuracy from 15% to 20% (+5 percentage points)
- However, this required **6.3x more inference time** (5.8 min → 36.6 min)
- **Trade-off:** One additional correct answer required ~31 extra minutes of computation

**2. Inference-Time Scaling Cannot Create Reasoning Skills**
- The reasoning model generated substantially longer reasoning traces (~493 tokens average)
- Despite longer traces, accuracy **remained at 15%** on single-sample evaluation
- This demonstrates that **inference-time techniques amplify existing capabilities but cannot create new reasoning abilities**

**3. Prompt Templates Have Major Impact**
- Different prompt formats showed significant variations in results
- The standard format used achieved 15-20% accuracy
- Alternative templates in the notebook showed different performance characteristics
- **Implication:** Prompt engineering is critical but not sufficient without underlying model capability

### Lessons for Post-Training

These experiments highlight a fundamental limitation:
> **Inference-time scaling is not a substitute for training-time reasoning improvement.**

The techniques explored (CoT, self-consistency, temperature scaling) work by:
- Amplifying existing problem-solving distributions in the model
- Taking majority votes over multiple attempts
- Encouraging more explicit reasoning

But they **cannot**:
- Create reasoning skills not already in the model
- Overcome fundamental capability gaps
- Solve problems the model couldn't solve in any single attempt

**For meaningful improvements in mathematical reasoning, focus should be on post-training methods:**
- Supervised fine-tuning on curated reasoning examples
- Preference optimization (DPO/ORPO/SimPO)
- Reinforcement learning with mathematical correctness rewards
- Reasoning-specific architectural modifications

---

## Core Areas

* Supervised Fine-Tuning (SFT)
* Preference Optimization (DPO / ORPO / SimPO)
* RL-based post-training
* **Reasoning and test-time scaling** ⭐ (Primary Focus)
* Reward modeling and evaluation
* Dataset construction and filtering
* Scalable training and inference systems
* Open-weight reasoning model experimentation

---

## Long-Term Vision

This project aims to evolve into a modular end-to-end post-training framework for reasoning-centric language models, covering:

* Data pipelines
* Training infrastructure
* Alignment and preference optimization
* RLHF-style workflows
* Evaluation harnesses
* Inference and serving
* Agentic and reasoning-oriented experimentation

## Design Principles

* Research-first implementations
* Reproducibility over hype
* Minimal abstractions where possible
* Clear experimental structure
* Systems-level understanding of the full stack
* Open collaboration and transparent iteration

## Planned Integrations

* Hugging Face Transformers
* TRL
* vLLM
* SGLang
* Ray
* DeepSpeed / FSDP

## Repository Goals

* Implement core post-training algorithms from first principles
* Reproduce influential papers and training recipes
* Build scalable experimentation workflows
* Develop intuition for reasoning and alignment systems

---

## 📁 Repository Structure & Navigation Guide

This section guides new users through the repository structure and how to navigate the project.

### **1. `base_model/` - Custom Model Architecture**
**What it does:** This folder contains the custom architecture code for building your own language model from scratch.

**Key workflow:**
- Download pre-trained weights from Hugging Face (via the `downloading_the_base_model/` folder)
- Define your custom model architecture
- Load the downloaded weights into your custom architecture
- Use this as the foundation for all downstream tasks (fine-tuning, inference, evaluation)

**Contains:** Architecture definitions, model initialization code, and weight loading utilities
- `qwen.py` - Custom Qwen model implementation
- `qwen.ipynb` - Jupyter notebook demonstrating model architecture

**When to use:** Start here if you want to understand how to build custom architectures and load pre-trained weights

👉 See [base_model/README.md](base_model/README.md) for detailed documentation

---

### **2. `downloading_the_base_model/` - Model Download Pipeline**
**What it does:** Handles downloading pre-trained models from Hugging Face Hub and managing model versioning.

**Key workflow:**
- Download models directly from Hugging Face
- Cache models locally for efficient reuse
- Handle model metadata and configuration files
- Prepare models for use in custom architectures

**Contains:** Download utilities, caching mechanisms, and model fetching scripts
- `download_model.py` - Main script for downloading models from Hugging Face Hub

**When to use:** Run this first in your workflow to download the base model weights before building custom architectures

**Example:** Download a Qwen model before loading it into your custom architecture

👉 See [downloading_the_base_model/README.md](downloading_the_base_model/README.md) for detailed documentation

---

### **3. `evaluating_reasoning_models/` - Model Evaluation & Benchmarking**
**What it does:** Comprehensive evaluation framework for assessing reasoning capabilities of language models.

**Key workflow:**
- Load your model and tokenizer
- Evaluate on reasoning benchmarks (e.g., MATH500)
- Measure accuracy, correctness, and reasoning quality
- Generate evaluation reports and statistics
- Compare inference strategies (greedy, self-consistency, etc.)

**Contains:** Evaluation scripts, benchmark datasets, and metrics calculation
- `math500_test.json` - MATH500 benchmark test set
- `evaluating_reasoning_models.py` - Core evaluation logic
- `model_and_tokenizer.py` - Model loading utilities
- `text_generation_wrapper.py` - Generation wrapper for inference
- `load_math_500.py` - Dataset loading utilities

**When to use:** After fine-tuning or before deploying your model to measure reasoning performance

**Typical workflow:** Generate predictions → Compare with ground truth → Calculate metrics

👉 See [evaluating_reasoning_models/README.md](evaluating_reasoning_models/README.md) for detailed documentation

---

### **4. `generating_text_with_pre_trained_llm/` - Text Generation & Inference**
**What it does:** Scripts and utilities for generating text using pre-trained language models with various decoding strategies.

**Key workflow:**
- Load a pre-trained model (or your custom model)
- Generate text with configurable parameters
- Implement different decoding strategies (greedy, top-k, nucleus sampling, etc.)
- Collect generation statistics and performance metrics
- Support batch inference for efficiency

**Contains:** Generation scripts, inference utilities, and statistics collection
- `generate.py` - Main text generation script
- `generate_stats.py` - Statistics collection from generations

**When to use:** For inference, testing, and generating predictions on new prompts

**Examples:**
- Generate solutions to math problems
- Test model behavior with different sampling strategies
- Collect text for evaluation

👉 See [generating_text_with_pre_trained_llm/README.md](generating_text_with_pre_trained_llm/README.md) for detailed documentation

---

### **5. `improving_reasoning_with_inference_time_scaling/` - Inference-Time Optimization (⭐ Primary Focus)**
**What it does:** Advanced inference-time techniques to enhance model reasoning without retraining.

**Key workflow:**
- Apply temperature scaling, nucleus sampling, and other techniques
- Implement Chain-of-Thought (CoT) prompting
- Generate multiple reasoning paths and take majority vote (self-consistency)
- Measure improvements in reasoning accuracy
- Benchmark against baseline generation strategies

**Contains:** Inference optimization scripts, evaluation datasets, and results
- `improving_reasoning_with_inference_time_scaling.py` - Core implementation
- `math500-base-cuda-cot-greedy.jsonl` - Results with CoT greedy decoding
- `math500-base-cuda-cot-self-consistency.jsonl` - Results with CoT + self-consistency

**When to use:** To improve reasoning performance of existing models at inference time without retraining

**Key insight:** Different inference strategies can significantly boost reasoning accuracy—explore and compare!

👉 See [improving_reasoning_with_inference_time_scaling/README.md](improving_reasoning_with_inference_time_scaling/README.md) for detailed documentation

---

## 🚀 Quick Start Workflow

**For new users, follow this typical workflow:**

1. **Download a base model:**
   ```bash
   cd downloading_the_base_model
   python download_model.py  # Download from Hugging Face
   ```

2. **Define and initialize your custom architecture:**
   ```bash
   cd ../base_model
   # Review qwen.py to understand the architecture
   python qwen.py  # Initialize your model
   ```

3. **Generate text with your model:**
   ```bash
   cd ../generating_text_with_pre_trained_llm
   python generate.py  # Generate predictions
   ```

4. **Evaluate reasoning performance:**
   ```bash
   cd ../evaluating_reasoning_models
   python evaluating_reasoning_models.py  # Run evaluation on benchmarks
   ```

5. **Improve reasoning with inference-time scaling:**
   ```bash
   cd ../improving_reasoning_with_inference_time_scaling
   python improving_reasoning_with_inference_time_scaling.py  # Apply scaling techniques
   ```

---

## 📖 Recommended Reading Order

- **New to the repo?** Start with the main README (you're here!) and then explore each folder's README
- **Want to understand architecture?** → `base_model/README.md`
- **Need to download a model?** → `downloading_the_base_model/README.md`
- **Ready to generate text?** → `generating_text_with_pre_trained_llm/README.md`
- **Want to evaluate?** → `evaluating_reasoning_models/README.md`
- **Interested in improving reasoning?** → `improving_reasoning_with_inference_time_scaling/README.md` (⭐ focus area)
* Contribute back to the open-source AI ecosystem

## Status

Active early-stage development.
The repository will continuously evolve toward larger-scale post-training and reasoning system experiments.

Contributions, discussions, and research collaboration are welcome.
