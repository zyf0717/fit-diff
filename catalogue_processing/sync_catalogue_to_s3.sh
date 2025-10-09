#!/bin/bash

# S3 sync script for catalogue.csv file
# Syncs only the catalogue file to project-traco-benchmarking bucket with fit_files/ prefix

set -e  # Exit on any error

# Load environment variables from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Loaded configuration from .env file"
else
    echo "Warning: .env file not found, using default values"
fi

# Configuration
BUCKET="${S3_BUCKET:-project-traco-benchmarking}"
PREFIX="fit_files/"
AWS_PROFILE="${AWS_PROFILE:-project-traco}"
LOCAL_FILE="fit_files_catalogue.csv"

# Check if local catalogue file exists
if [ ! -f "$LOCAL_FILE" ]; then
    echo "Error: $LOCAL_FILE not found in current directory"
    exit 1
fi

echo "Syncing catalogue file to S3..."
echo "Source: $LOCAL_FILE"
echo "Destination: s3://$BUCKET/$PREFIX$LOCAL_FILE"
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: ${AWS_REGION:-not set}"
echo "----------------------------------------"

# Sync the catalogue file to S3
aws s3 cp "$LOCAL_FILE" "s3://$BUCKET/$PREFIX$LOCAL_FILE" --profile "$AWS_PROFILE"

if [ $? -eq 0 ]; then
    echo "Successfully synced $LOCAL_FILE to S3"
    echo "Available at: s3://$BUCKET/$PREFIX$LOCAL_FILE"
else
    echo "Failed to sync $LOCAL_FILE to S3"
    exit 1
fi