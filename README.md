# Open Post Training
*this repository will be structured properly in coming days, ...on-it⚠️

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

## Core Areas

* Supervised Fine-Tuning (SFT)
* Preference Optimization (DPO / ORPO / SimPO)
* RL-based post-training
* **Reasoning and test-time scaling** ⭐ (Primary Focus)
* Reward modeling and evaluation
* Dataset construction and filtering
* Scalable training and inference systems
* Open-weight reasoning model experimentation

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
* Contribute back to the open-source AI ecosystem

## Status

Active early-stage development.
The repository will continuously evolve toward larger-scale post-training and reasoning system experiments.

Contributions, discussions, and research collaboration are welcome.
