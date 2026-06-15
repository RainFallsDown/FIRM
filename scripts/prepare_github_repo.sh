#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-FIRM}"
GITHUB_USER="${2:-RainFallsDown}"
REMOTE_MODE="${3:-https}"

if [ -d .git ]; then
  echo "Local .git already exists in $(pwd)"
else
  git init -b main
fi

git lfs install

git add \
  .gitattributes \
  .gitignore \
  README.md \
  docs/ \
  firm_sim/ \
  scripts/ \
  tests/ \
  PYBULLET/ \
  tianqing_urdf.zip

echo
echo "Review staged files with: git status"
echo "Then commit with:"
echo "  git commit -m \"Add Genesis-based FIRM-Sim scenes\""

if [ "$REMOTE_MODE" = "ssh" ]; then
  REMOTE_URL="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
else
  REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

echo
echo "Suggested remote:"
echo "  git remote add origin ${REMOTE_URL}"
echo "  git push -u origin main"
