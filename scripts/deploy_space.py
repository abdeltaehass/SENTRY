"""Create and push the SENTRY Hugging Face Space (one command).

Vendors the `src/` tree + `configs/` into the Space (no install-from-git), so the
Space is self-contained.

Prerequisite: log in once with a WRITE token —
    hf auth login            # token from https://huggingface.co/settings/tokens

Then:
    python scripts/deploy_space.py --repo-id <your-username>/sentry-incident-reports
"""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi, create_repo


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy SENTRY to a HF Space")
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/sentry-incident-reports")
    ap.add_argument("--space-dir", default="deploy/space")
    args = ap.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    # app.py / requirements.txt / README.md at the Space root
    api.upload_folder(folder_path=args.space_dir, repo_id=args.repo_id, repo_type="space")
    # vendored source + config
    api.upload_folder(folder_path="src", path_in_repo="src", repo_id=args.repo_id,
                      repo_type="space", ignore_patterns=["**/__pycache__/*"])
    api.upload_folder(folder_path="configs", path_in_repo="configs", repo_id=args.repo_id,
                      repo_type="space")
    print(f"\nLive: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
