import asyncio
import logging
import os
import subprocess
from pathlib import Path

import aioboto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError


# Load .env file from root directory
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from .env with fallbacks
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_BUCKET = os.getenv("S3_BUCKET")

# Validate configuration
if not AWS_PROFILE or not S3_BUCKET:
    logger.error("Missing AWS configuration. Check .env file.")
    exit(1)


async def list_s3_fit_files(bucket_name, prefix, profile_name):
    """
    List all .fit files from S3 bucket using aioboto3
    """
    fit_files = []
    session = aioboto3.Session(profile_name=profile_name)

    try:
        async with session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            async for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".fit"):
                            # ETag is typically the MD5 hash, but remove quotes if present
                            etag = obj.get("ETag", "").strip('"')

                            fit_files.append(
                                {
                                    "key": key,
                                    "filename": key.split("/")[-1],
                                    "size_mb": round(obj["Size"] / (1024 * 1024), 2),
                                    "last_modified": obj["LastModified"],
                                    "etag": etag,  # MD5 hash from S3 ETag
                                }
                            )

    except (NoCredentialsError, ClientError) as e:
        logger.error(f"S3 error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []

    return sorted(fit_files, key=lambda x: x["key"])


def process_file_list(file_list):
    """
    Process the list of files into a DataFrame and add device_type and tags
    """
    df = pd.DataFrame(file_list)
    df["device_type"] = df["filename"].apply(
        lambda x: "test" if "pacer" in x.lower() else "ref"
    )
    df["tags"] = df["key"].apply(
        lambda x: [x.split("/")[1]] if len(x.split("/")) > 1 else []
    )
    return df[["etag", "key", "size_mb", "device_type", "tags"]]


async def main():
    logger.info(f"Updating catalogue from S3 bucket: {S3_BUCKET}")

    # Get current files from S3
    fit_files = await list_s3_fit_files(S3_BUCKET, "fit_files/", AWS_PROFILE)
    if not fit_files:
        logger.info("No .fit files found in S3.")
        return

    # Process into DataFrame
    latest_df = process_file_list(fit_files)
    logger.info(f"Found {len(latest_df)} files in S3")

    # Load existing catalogue (create empty if doesn't exist)
    catalogue_file = "fit_files_catalogue.csv"
    if os.path.exists(catalogue_file):
        existing_df = pd.read_csv(catalogue_file)
        logger.info(f"Loaded existing catalogue with {len(existing_df)} entries")
    else:
        existing_df = pd.DataFrame(
            columns=["etag", "key", "size_mb", "device_type", "tags"]
        )
        logger.info("No existing catalogue found, creating new one")

    # Find new files
    new_files = latest_df[~latest_df["etag"].isin(existing_df["etag"])]

    if new_files.empty:
        logger.info("No new files to add")
        return

    # Update catalogue
    updated_df = pd.concat([existing_df, new_files], ignore_index=True)
    updated_df.to_csv(catalogue_file, index=False)
    logger.info(f"Added {len(new_files)} new files to catalogue")

    # Sync to S3
    try:
        subprocess.run(
            ["./sync_catalogue_to_s3.sh"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Successfully synced catalogue to S3")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sync to S3: {e}")


if __name__ == "__main__":
    asyncio.run(main())
