# upload_results.py — Push results to HuggingFace results repo

import glob
import os
from huggingface_hub import HfApi
import config as cfg


def upload_results() -> None:
    # Read directly from environment — most reliable in GitHub Actions
    token   = os.environ.get("HF_TOKEN", "") or cfg.HF_TOKEN
    repo_id = os.environ.get("HF_RESULTS_REPO", "") or cfg.HF_RESULTS_REPO

    if not token:
        raise ValueError("HF_TOKEN is not set.")
    if not repo_id:
        raise ValueError("HF_RESULTS_REPO is not set.")

    print(f"Uploading to: {repo_id}")
    print(f"Token present: {bool(token)}, length: {len(token)}")

    result_files = glob.glob(os.path.join(cfg.RESULTS_DIR, "*.json"))
    if not result_files:
        print("No result files found in results/")
        return

    api = HfApi(token=token)

    for f in result_files:
        repo_path = f"results/{os.path.basename(f)}"
        print(f"  Uploading {os.path.basename(f)} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"[auto] Update PCMCI+ results",
        )

    print(f"Pushed {len(result_files)} file(s) to {repo_id}")


if __name__ == "__main__":
    upload_results()
