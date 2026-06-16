import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local checkpoint file to a Hugging Face model repo."
    )
    parser.add_argument(
        "--repo-id",
        default="devshaheen/qwen3.5_rlvr_grpo_run_33_steps",
        help="Hugging Face repo ID to upload to.",
    )
    parser.add_argument(
        "--model-file",
        default="qwen3-0.6B-rlvr-grpo-step00033-interrupt.pth",
        help="Local checkpoint file path relative to this script.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path inside the repo. Defaults to the local file name.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, uses logged-in token or HF_TOKEN env var.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_path = root / args.model_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Place the checkpoint in the same folder as this script or provide --model-file."
        )

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
        token=args.token,
    )

    uploaded_path = args.path_in_repo or model_path.name
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=uploaded_path,
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token,
    )

    print(f"Uploaded {model_path.name} to huggingface.co/{args.repo_id}/{uploaded_path}")


if __name__ == "__main__":
    main()
