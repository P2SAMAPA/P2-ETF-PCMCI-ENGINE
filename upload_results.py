# upload_results.py — Push results to HuggingFace results repo

import glob
import os
from huggingface_hub import HfApi, CommitOperationAdd
import config as cfg

api       = HfApi(token=cfg.HF_TOKEN)
REPO_ID   = cfg.HF_RESULTS_REPO
REPO_TYPE = "dataset"


def upload_results() -> None:
    result_files = glob.glob(os.path.join(cfg.RESULTS_DIR, "*.json"))

    if not result_files:
        print("No result files found.")
        return

    operations = [
        CommitOperationAdd(
            path_in_repo=f"results/{os.path.basename(f)}",
            path_or_fileobj=f,
        )
        for f in result_files if os.path.exists(f)
    ]

    if not operations:
        return

    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE,
                    exist_ok=True, private=False)
    api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message="[auto] Update PCMCI+ signals and results",
    )
    print(f"Pushed {len(operations)} file(s) to {REPO_ID}")


if __name__ == "__main__":
    upload_results()
