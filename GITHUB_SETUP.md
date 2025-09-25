# 🚀 GitHub Setup Guide for LEAP

This guide will help you push the LEAP implementation to GitHub.

## Prerequisites

### 1. Install Git (if not already installed)

**Windows:**
- Download Git from: https://git-scm.com/download/windows
- Run the installer with default settings
- Restart your terminal/command prompt

**macOS:**
```bash
# Using Homebrew
brew install git

# Or download from: https://git-scm.com/download/mac
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# CentOS/RHEL
sudo yum install git
```

### 2. Configure Git (first time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step-by-Step GitHub Setup

### Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click the "+" icon → "New repository"
3. Repository name: `LEAP_Code` or `LEAP-MoE-Optimization`
4. Description: `LEAP: Learning Expert Adaptation & Pruning for Task-Specialized MoE Language Models`
5. Choose "Public" (recommended for open source)
6. **Don't** initialize with README (we already have one)
7. Click "Create repository"

### Step 2: Initialize and Push Repository

Open terminal/command prompt in the LEAP_Code directory and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete LEAP implementation

- Implement Meta-RL based expert pruning agent
- Add RL-based routing adaptation with active learning
- Support for Llama 4 Maverick (17Bx128E) and Qwen3-235B-A22B
- Complete evaluation framework with task-specific metrics
- CLI interface and comprehensive examples
- Achieve 87.5% parameter reduction with 95-98% performance retention"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/LEAP_Code.git

# Push to GitHub
git push -u origin main
```

### Step 3: Verify Upload

1. Go to your GitHub repository
2. Verify all files are uploaded
3. Check that README.md displays correctly

## Alternative: GitHub Desktop

If you prefer a GUI:

1. Download GitHub Desktop: https://desktop.github.com/
2. Install and sign in to your GitHub account
3. Click "Add an Existing Repository from your Hard Drive"
4. Select the LEAP_Code folder
5. Click "Publish repository" 
6. Choose repository name and visibility
7. Click "Publish Repository"

## Repository Structure Verification

After pushing, your repository should contain:

```
LEAP_Code/
├── README.md                 ✅ Project documentation
├── setup.py                  ✅ Package installation
├── requirements.txt          ✅ Dependencies
├── LICENSE                   ✅ MIT License
├── .gitignore               ✅ Git ignore rules
├── CONTRIBUTING.md          ✅ Contribution guide
├── CHANGELOG.md             ✅ Version history
├── GITHUB_SETUP.md          ✅ This setup guide
├── sample_paper.pdf         ✅ Original research paper
├── leap/                    ✅ Main package
├── configs/                 ✅ Configuration files
└── examples/                ✅ Usage examples
```

## Post-Upload Steps

### 1. Add Repository Description and Topics

In your GitHub repository:
1. Click the gear icon next to "About"
2. Add description: "LEAP: Learning Expert Adaptation & Pruning for Task-Specialized MoE Language Models"
3. Add topics: `machine-learning`, `mixture-of-experts`, `model-compression`, `reinforcement-learning`, `pytorch`, `llama`, `qwen`, `expert-pruning`, `active-learning`, `moe`
4. Add website (if you have one)

### 2. Enable GitHub Pages (Optional)

To create a project website:
1. Go to repository Settings
2. Scroll to "Pages"
3. Source: "Deploy from a branch"
4. Branch: "main", folder: "/ (root)"
5. Your documentation will be available at: `https://yourusername.github.io/LEAP_Code/`

### 3. Add Badges to README

Add these badges to the top of your README.md:

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/LEAP_Code.svg)](https://github.com/yourusername/LEAP_Code/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/yourusername/LEAP_Code.svg)](https://github.com/yourusername/LEAP_Code/network)
```

### 4. Create Release

1. Go to "Releases" → "Create a new release"
2. Tag version: `v0.1.0`
3. Release title: `LEAP v0.1.0 - Initial Release`
4. Description: Copy from CHANGELOG.md
5. Click "Publish release"

## Troubleshooting

### Large File Issues
If you get errors about large files:
```bash
# Remove large files from tracking
git rm --cached path/to/large/file
echo "path/to/large/file" >> .gitignore
git add .gitignore
git commit -m "Remove large files"
```

### Authentication Issues
If you get authentication errors:
1. Use personal access token instead of password
2. Go to GitHub Settings → Developer settings → Personal access tokens
3. Generate new token with repo permissions
4. Use token as password when prompted

### Repository Already Exists
If you get "repository already exists" error:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
```

## Next Steps After Upload

1. **Star your own repository** (helps with visibility)
2. **Share with the community** (Reddit, Twitter, LinkedIn)
3. **Submit to awesome lists** (awesome-pytorch, awesome-transformers)
4. **Write a blog post** about your implementation
5. **Create issues** for future enhancements
6. **Set up CI/CD** with GitHub Actions

## Contact

If you need help with the setup:
1. Create an issue in the repository
2. Check GitHub's documentation: https://docs.github.com/
3. Use GitHub's community forum: https://github.community/

---

🎉 **Congratulations!** Your LEAP implementation is now on GitHub and ready for the world to use!
