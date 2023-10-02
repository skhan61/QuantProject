#!/bin/bash

# Navigate to your project repository
cd "/home/sayem/Desktop/Project"

# Explicitly add the 'Old' folder recursively
git add Old/

# Check for changes in both working directory and staged area
if [ -n "$(git diff)" ] || [ -n "$(git diff --cached)" ]; then
    # Add all other changes not mentioned in .gitignore
    git add .

    # Commit the changes
    git commit -m "Automatic commit at $(date)"

    # Determine which branch to push based on the current active branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    if [ "$CURRENT_BRANCH" = "main" ]; then
        git push git@github.com:skhan61/QuantProject.git main
    else
        echo "You are currently on the $CURRENT_BRANCH branch. Please switch to the main branch to auto-push."
    fi
else
    echo "No changes detected."
fi
