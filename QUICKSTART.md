# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Prerequisites
- ✅ Anaconda or Miniconda installed
- ✅ Git installed

---

## Setup (One-Time)

### 1. Create Environment

```bash
conda env create -f environment.yml
conda activate aidj
```

### 2. Download Dataset

1. Go to: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
2. Download dataset (~5GB)
3. Extract to `data/raw/`

### 3. Get Spotify API Keys

1. Go to: https://developer.spotify.com/dashboard
2. Create app → Get Client ID & Secret
3. Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your credentials
```

---

## Daily Workflow

### Mac

```bash
# Activate environment
conda activate aidj

# Start coding
jupyter notebook

# Commit changes
git add .
git commit -m "Your message"
git push
```

### Windows

```powershell
# Pull latest
git pull

# Activate environment
conda activate aidj

# Start coding
jupyter notebook
```

---

## That's It!

📖 For detailed instructions: [SETUP.md](SETUP.md)

📊 For project roadmap: [AI_DJ_Project_Plan.md](AI_DJ_Project_Plan.md)

💻 Main notebook: [notebooks/ai_dj_main.ipynb](notebooks/ai_dj_main.ipynb)
