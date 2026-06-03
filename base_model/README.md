# Base Model Architecture

## Overview

This folder contains the custom model architecture definitions for building your own language model from scratch. The core idea is to:

1. **Define your own architecture** - Create custom model classes with desired properties
2. **Load pre-trained weights** - Download weights from Hugging Face using the weights from `downloading_the_base_model/`
3. **Initialize the model** - Combine your architecture with pre-trained weights
4. **Use as foundation** - Deploy for inference, fine-tuning, or evaluation

## Contents

### `qwen.py`
Main file containing the custom Qwen model architecture implementation.

**Key components:**
- Custom model class definitions
- Architecture-specific forward passes
- Weight loading utilities
- Tokenizer integration

**Usage:**
```python
from qwen import YourCustomModel
model = YourCustomModel.from_pretrained("qwen/model-name")
```

### `qwen.ipynb`
Jupyter notebook with step-by-step demonstrations of:
- Building custom model architectures
- Loading pre-trained weights
- Model initialization and configuration
- Testing the model architecture

**Best for:** Understanding the architecture interactively and experimenting with different configurations

## Workflow

### Step 1: Define Architecture
In your Python file, define your custom model class:
```python
class CustomQwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers and components
        self.transformer = TransformerStack(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        # Forward pass logic
        ...
```

### Step 2: Load Pre-trained Weights
```python
# First, download weights using downloading_the_base_model/
# Then load them into your custom architecture
model = CustomQwenModel(config)
model.load_state_dict(torch.load("path/to/weights.pt"))
```

### Step 3: Initialize Model
```python
# Complete model initialization with tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("qwen/model-name")
model = model.to("cuda")
model.eval()
```

### Step 4: Next Steps
Your initialized model is ready for:
- **Text Generation** → Use `../generating_text_with_pre_trained_llm/`
- **Evaluation** → Use `../evaluating_reasoning_models/`
- **Fine-tuning** → Create your training loop or use training frameworks
- **Inference-time scaling** → Use `../improving_reasoning_with_inference_time_scaling/`

## Key Concepts

### Why Custom Architecture?
- **Control:** Customize model components for your specific needs
- **Experimentation:** Test architectural modifications
- **Reproducibility:** Define exact architecture used in your work
- **Integration:** Combine with pre-trained weights for rapid development

### Architecture vs. Weights
- **Architecture:** The structural definition of how data flows (qwen.py)
- **Weights:** The learned parameters from pre-training (downloaded separately)
- **Combined:** Architecture + weights = fully initialized model ready for use

### Model Loading Pattern
```
Download Weights (downloading_the_base_model/)
         ↓
Define Custom Architecture (base_model/)
         ↓
Load Weights into Architecture
         ↓
Initialize with Tokenizer
         ↓
Ready for Inference/Training
```

## Configuration

### Common Configuration Parameters
- `hidden_size` - Dimension of transformer hidden states
- `num_hidden_layers` - Number of transformer blocks
- `num_attention_heads` - Number of attention heads
- `vocab_size` - Size of vocabulary
- `max_position_embeddings` - Maximum sequence length
- `intermediate_size` - Dimension of feedforward layer

Example:
```python
config = {
    "hidden_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "vocab_size": 151936,  # Qwen specific
    "max_position_embeddings": 2048,
}
```

## Common Issues & Solutions

### Issue: Shape Mismatch When Loading Weights
**Problem:** Downloaded weights don't match your architecture
**Solution:** Ensure your model config matches the original model's config

### Issue: Out of Memory
**Problem:** Model too large for your GPU
**Solution:**
- Use smaller model variant
- Enable gradient checkpointing
- Use quantization techniques
- Reduce batch size

### Issue: Tokenizer Mismatch
**Problem:** Downloaded tokenizer doesn't match model weights
**Solution:** Use the tokenizer from the same model source

## Best Practices

1. **Document your architecture** - Add comments explaining non-standard choices
2. **Test weight loading** - Verify shapes match before full training runs
3. **Version your architecture** - Keep track of architectural changes
4. **Use config files** - Separate architecture definition from configuration
5. **Test inference** - Do a quick forward pass after initialization

## Useful Resources

- [Hugging Face Model Hub](https://huggingface.co/models) - Browse available models
- [PyTorch nn.Module](https://pytorch.org/docs/stable/nn.html) - Base class documentation
- Model-specific documentation - Qwen, LLaMA, Mistral, etc.

## Next Steps

After setting up your architecture:
1. Load it with weights from `downloading_the_base_model/`
2. Test generation in `generating_text_with_pre_trained_llm/`
3. Evaluate performance in `evaluating_reasoning_models/`
4. Optimize reasoning in `improving_reasoning_with_inference_time_scaling/`

## Questions?

Refer to the notebooks for working examples, or check parent README for project-wide guidance.
