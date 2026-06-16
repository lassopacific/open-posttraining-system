from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="training_reasoning_models_with_reinforcement_learning", repo_id="devshaheen/qwen3.5_rlvr_grpo_run_33_steps", repo_type="model")
