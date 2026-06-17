from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent  # Get parent of current directory
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from downloading_the_base_model.download_model import download_model

download_model("devshaheen/qwen3.5_0.6B_rlvr_grpo_run_33_steps", "qwen")  ## this is change later


import torch
from safetensors.torch import load_file

from base_model.qwen import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    load_hf_weights_into_qwen,
)

model_dir = Path.cwd() / "qwen"


# def main(prompt):
def load_model_and_tokenizer(which_model, use_compile):

    if which_model == "base":
        tokenizer = Qwen3Tokenizer(
            model_dir / "tokenizer.json",
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=False,
        )

    elif which_model == "reasoning":
        tokenizer = Qwen3Tokenizer(
            model_dir / "tokenizer.json",
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError("Not a valid model type")

    weights = load_file(model_dir / "model.safetensors")

    model = Qwen3Model(QWEN_CONFIG_06_B)

    load_hf_weights_into_qwen(
        model,
        param_config={
            "n_layers": QWEN_CONFIG_06_B["n_layers"],
            "hidden_dim": QWEN_CONFIG_06_B["hidden_dim"],
        },
        params=weights,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.to(torch.bfloat16)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer