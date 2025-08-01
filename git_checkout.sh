#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <branch-name>"
  exit 1
fi

BRANCH="$1"

git fetch origin

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠️ Uncommitted changes detected. Please commit or stash before switching branches."
  exit 1
fi

if git show-ref --quiet refs/heads/"$BRANCH"; then
  echo "Local branch '$BRANCH' already exists. Checking it out..."
  git checkout "$BRANCH"
else
  echo "Creating and checking out local branch '$BRANCH' from origin/$BRANCH..."
  git checkout -b "$BRANCH" "origin/$BRANCH"
fi