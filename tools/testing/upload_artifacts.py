import glob
import os
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def zip_artifacts() -> str:
    """Zip up the artifacts."""
    file_suffix = os.environ.get("ARTIFACTS_FILE_SUFFIX")
    if not file_suffix:
        raise ValueError("ARTIFACTS_FILE_SUFFIX is not set")
    file_name = f"{REPO_ROOT}/test-reports-{file_suffix}.zip"

    with zipfile.ZipFile(file_name, "w") as f:
        for file in glob.glob(f"{REPO_ROOT}/test/**/*.xml", recursive=True):
            f.write(file, os.path.relpath(file, REPO_ROOT))
        for file in glob.glob(f"{REPO_ROOT}/test/**/*.csv", recursive=True):
            f.write(file, os.path.relpath(file, REPO_ROOT))

    return file_name


def upload_to_s3_artifacts(file_name: str) -> None:
    """Upload the file to S3."""
    workflow_id = os.environ.get("GITHUB_RUN_ID")
    if not workflow_id:
        raise ValueError("GITHUB_RUN_ID is not set")
    import boto3  # type: ignore[import]

    S3_RESOURCE = boto3.client("s3")
    S3_RESOURCE.upload_file(
        file_name,
        "gha-artifacts",
        f"pytorch/pytorch/{workflow_id}/{Path(file_name).name}",
    )
    S3_RESOURCE.put_object(
        Body=b'',
        Bucket="gha-artifacts",
        Key=f"catttest_deleteme/{workflow_id}.txt",
    )

def zip_and_upload_artifacts() -> None:
    file_name = zip_artifacts()
    upload_to_s3_artifacts(file_name)


def trigger_upload_test_stats_intermediate_workflow() -> None:
    print("Triggering upload_test_stats_intermediate workflow")
    x = requests.post(
        "https://api.github.com/repos/pytorch/pytorch/actions/workflows/upload_test_stats_intermediate.yml/dispatches",
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        },
        json={
            "ref": "main",
            "inputs": {
                "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
                "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            },
        },
    )
    print(x.text)
