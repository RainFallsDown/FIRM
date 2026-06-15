# GitHub Upload Guide

This repo is already prepared for GitHub upload, with:

- `.gitignore` for local caches, videos, screenshots, and extracted robot assets
- `.gitattributes` for `tianqing_urdf.zip` via Git LFS
- a GitHub-friendly `README.md`
- a dedicated upload helper script: `scripts/prepare_github_repo.sh`

## Current Local Path

```text
/Users/chinchilla/Documents/New project 2/NIPS26
```

## Important Before Upload

### 1. Robot asset is large

`tianqing_urdf.zip` is stored through **Git LFS**.

Run:

```bash
git lfs install
```

### 2. Do not upload confidential paper copies

Local manuscript PDFs are reference material only.

Avoid publicly uploading:

- reviewer-copy manuscript PDFs
- translated PDFs
- dual-language PDFs

## Target GitHub Repo

Upstream repository:

```text
https://github.com/RainFallsDown/FIRM
```

Contributor account:

```text
ChinChilla-HTL
```

## Recommended Upload Flow

This folder is currently nested inside a larger local Git workspace, so for GitHub upload you should first initialize a **standalone repo inside `NIPS26`**.

### Option A: push directly to upstream

Use this when `ChinChilla-HTL` already has write access to `RainFallsDown/FIRM`.

```bash
git init -b main
git lfs install
git add .gitattributes .gitignore README.md docs/ firm_sim/ scripts/ tests/ PYBULLET/ tianqing_urdf.zip
git status
git commit -m "Add Genesis-based FIRM-Sim scenes"
git remote add origin https://github.com/RainFallsDown/FIRM.git
git push -u origin main
```

### Option B: push to your fork, then open a PR

Use this when you prefer keeping your own branch history under `ChinChilla-HTL`.

1. Create a fork on GitHub:

```text
https://github.com/ChinChilla-HTL/FIRM
```

2. Push locally to the fork:

```bash
git init -b main
git lfs install
git add .gitattributes .gitignore README.md docs/ firm_sim/ scripts/ tests/ PYBULLET/ tianqing_urdf.zip
git status
git commit -m "Add Genesis-based FIRM-Sim scenes"
git remote add origin https://github.com/ChinChilla-HTL/FIRM.git
git push -u origin main
```

3. Then open a pull request from:

```text
ChinChilla-HTL/FIRM:main  ->  RainFallsDown/FIRM:main
```

### SSH variant

```bash
git init -b main
git lfs install
git add .gitattributes .gitignore README.md docs/ firm_sim/ scripts/ tests/ PYBULLET/ tianqing_urdf.zip
git status
git commit -m "Add Genesis-based FIRM-Sim scenes"
git remote add origin git@github.com:RainFallsDown/FIRM.git
git push -u origin main
```

## Files You Probably Want to Keep Out of the Commit

Unless you explicitly want them in the repo, avoid staging:

- `588_FIRM_A_Benchmark_for_Indus.pdf`
- any local translated manuscript PDFs
- `outputs/`
- local `.mp4` recordings

## Final Sanity Check

Before pushing, confirm:

```bash
git status
git lfs ls-files
```

You want to see:

- source code changes
- docs
- `tianqing_urdf.zip` under LFS
- no local screenshots or videos
