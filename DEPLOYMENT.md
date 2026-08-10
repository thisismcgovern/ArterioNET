# ArterioNet Deployment Guide

Complete instructions for deploying ArterioNet to GitHub and PyPI.

---

## Step 1: Create GitHub Repository

### A. Initialize Local Git

```bash
cd /path/to/arterionet
git init
git add .
git commit -m "Initial commit: ArterioNet v0.1.0 with Y-NET dual-encoder architecture"
```

### B. Create Remote Repository

1. Go to https://github.com/new
2. **Repository name:** `arterionet`
3. **Description:** "Cuffless arterial blood pressure reconstruction from ECG and PPG"
4. **Visibility:** Public
5. **Initialize with:** None (already have files)
6. Click **Create repository**

### C. Push to GitHub

```bash
git remote add origin https://github.com/mcgovern-twumasi/arterionet.git
git branch -M main
git push -u origin main
```

---

## Step 2: GitHub Setup

### Add Collaborators

1. Go to `Settings` → `Collaborators`
2. Add: `Emefa Abena Apedo`

### Enable Discussions

1. `Settings` → `Features` → Check "Discussions"

### Create Release

1. Go to `Releases` → `Create a new release`
2. **Tag version:** `v0.1.0`
3. **Title:** `ArterioNet v0.1.0 — Initial Release`
4. **Description:**
   ```
   ### Features
   - Y-NET dual-encoder U-Net for ABP reconstruction
   - ISO 81060-3:2022 compliant DBP prediction
   - MLX optimization for Apple Silicon (4-5× faster)
   - Clinical biomarkers (ASI, BPV)
   - AAMI compliance validation
   
   ### Performance
   - Waveform r: 0.984 (mean), 0.986 (median)
   - DBP: ME=-2.21 mmHg, SD=5.37 mmHg (AAMI pass)
   - Inference: 8-12 ms on M1 (MLX), 50 ms on CPU
   
   ### Authors
   - McGovern Twumasi Owusu-Bekoe
   - Emefa Abena Apedo
   - Supervised by: Ing. Dr. Isaac Acquah (KNUST)
   ```
5. Click **Publish release**

---

## Step 3: PyPI Setup

### A. Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Register account (e.g., `mcgovern-twumasi`)
3. Add email verification
4. Enable 2FA (Security > Account settings > Two-factor authentication)

### B. Create API Token

1. Go to `Account settings` → `API tokens`
2. Click **Add API token**
3. **Scope:** Entire account
4. **Name:** `github-action-token`
5. Copy token (save safely)

### C. Add GitHub Secrets

1. Go to GitHub repo → `Settings` → `Secrets and variables` → `Actions`
2. Click **New repository secret**
3. **Name:** `PYPI_API_TOKEN`
4. **Value:** Paste the token from Step B
5. Click **Add secret**

---

## Step 4: Build and Publish

### Build Locally (Test)

```bash
pip install build twine
python -m build
twine check dist/*
```

### Upload to Test PyPI

```bash
twine upload --repository testpypi dist/* --username __token__ --password pypi-AgEIcHlwaS5vcmc...
```

### Install from Test PyPI

```bash
pip install --index-url https://test.pypi.org/simple/ arterionet
```

### Publish to PyPI

```bash
twine upload dist/* --username __token__ --password pypi-AgEIcHlwaS5vcmc...
```

Or automatic via GitHub Actions: Push a tag and it auto-publishes.

---

## Step 5: Verify Installation

```bash
pip install arterionet
python -c "import arterionet; print(f'ArterioNet {arterionet.__version__} loaded')"
```

---

## Step 6: CI/CD Automation

GitHub Actions automatically:
- ✓ Runs tests on every push
- ✓ Formats code check (Black)
- ✓ Uploads coverage to Codecov
- ✓ Publishes to PyPI on release

### Trigger Workflow

To publish:
```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will:
1. Run tests
2. Build package
3. Publish to PyPI automatically

---

## Step 7: Documentation (ReadTheDocs)

### Setup RTD

1. Go to https://readthedocs.org/accounts/signup/
2. Sign up with GitHub
3. Click **Import a Project**
4. Select `arterionet` repository
5. Click **Next**
6. Keep defaults, click **Build**

### Create `docs/` Directory (Optional)

```bash
mkdir -p docs
pip install sphinx
sphinx-quickstart docs
```

---

## Step 8: Publish on Zenodo (for Citation)

1. Go to https://zenodo.org/
2. Sign in with GitHub
3. Click username → **GitHub**
4. Flip switch for `arterionet`
5. Create a release on GitHub
6. Zenodo auto-captures it and generates DOI

---

## Step 9: Add Badges to README

In `README.md`:

```markdown
[![PyPI version](https://badge.fury.io/py/arterionet.svg)](https://badge.fury.io/py/arterionet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/mcgovern-twumasi/arterionet.svg)](https://github.com/mcgovern-twumasi/arterionet/releases)
[![DOI](https://zenodo.org/badge/[RECORD-ID].svg)](https://zenodo.org/badge/latestdoi/[RECORD-ID])
```

---

## Checklist

- [ ] Local git initialized
- [ ] GitHub repository created
- [ ] Pushed to main branch
- [ ] Added collaborators
- [ ] PyPI account created
- [ ] API token generated
- [ ] GitHub secret configured
- [ ] Built and tested locally
- [ ] Published to PyPI
- [ ] Installed and verified
- [ ] GitHub Actions working
- [ ] ReadTheDocs set up
- [ ] Zenodo DOI captured
- [ ] Badges added to README

---

## Future Updates

To update the package:

```bash
# Update version in pyproject.toml
# Commit changes
git add .
git commit -m "Update to v0.2.0"

# Create release
git tag v0.2.0
git push origin v0.2.0

# GitHub Actions auto-publishes to PyPI
```

---

## Support

For deployment issues:
- Check GitHub Actions logs
- Review PyPI upload docs: https://packaging.python.org/tutorials/packaging-projects/
- Contact: mcgovernowusubekoe@gmail.com

---

**Package:** ArterioNet v0.1.0  
**Authors:** McGovern Twumasi Owusu-Bekoe, Emefa Abena Apedo  
**Last Updated:** 2024-08-09
