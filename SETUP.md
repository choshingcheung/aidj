# AI DJ Project - Setup Guide

This guide covers setup for **both Mac and Windows** using Anaconda/Miniconda.

## Prerequisites

### Install Conda (if not already installed)

**Mac:**
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**Windows:**
- Download Anaconda from: https://www.anaconda.com/download
- Or Miniconda from: https://docs.conda.io/en/latest/miniconda.html
- Run the installer

---

## Quick Setup (Recommended)

### Step 1: Clone/Navigate to Project

**Mac:**
```bash
cd ~/codespace/aidj
```

**Windows (PowerShell):**
```powershell
cd "C:\vscode workspace\aidj\aidj"
```

### Step 2: Create Conda Environment

```bash
# Create environment from file (works on both Mac and Windows)
conda env create -f environment.yml

# Activate the environment
conda activate aidj
```

That's it! Jump to "Download Dataset" section below.

---

## Manual Setup (Alternative)

If the environment.yml method doesn't work:

### Step 1: Create Environment

```bash
# Create environment with Python 3.11
conda create -n aidj python=3.11 -y

# Activate it
conda activate aidj
```

### Step 2: Install Core Packages

```bash
# Install from conda-forge (includes LightFM!)
conda install -c conda-forge \
  numpy pandas scipy \
  matplotlib seaborn plotly \
  scikit-learn xgboost \
  lightfm \
  librosa soundfile \
  jupyter notebook ipywidgets \
  tqdm requests python-dotenv \
  spotipy -y
```

### Step 3: Install Pip Packages

```bash
# Install remaining packages via pip
pip install pydub
```

### Step 4: Verify Installation

**Mac:**
```bash
python src/utils/config.py
```

**Windows:**
```powershell
python src\utils\config.py
```

---

## Download Spotify Million Playlist Dataset

### Step 1: Get Dataset

1. Go to [AICrowd Spotify Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
2. Create free account / log in
3. Click "Download Dataset"
4. Download `spotify_million_playlist_dataset.zip` (~5GB)

### Step 2: Extract to Project

**Mac:**
```bash
# Extract to data/raw/
unzip spotify_million_playlist_dataset.zip -d data/raw/
```

**Windows (PowerShell):**
```powershell
# Extract using Windows built-in or 7-Zip
Expand-Archive -Path spotify_million_playlist_dataset.zip -DestinationPath data\raw\
```

**Expected structure:**
```
data/raw/
├── data/
│   ├── mpd.slice.0-999.json
│   ├── mpd.slice.1000-1999.json
│   └── ...
└── README.md
```

---

## Set Up Spotify API Credentials

### Step 1: Create Spotify Developer App

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with Spotify account (create one if needed)
3. Click "Create an App"
   - Name: "AI DJ Project"
   - Description: "Academic project for playlist generation"
4. Click "Settings"
5. Copy your **Client ID** and **Client Secret**

### Step 2: Create .env File

**Mac:**
```bash
# Copy example file
cp .env.example .env

# Edit with your favorite editor
nano .env
# or
code .env
```

**Windows (PowerShell):**
```powershell
# Copy example file
copy .env.example .env

# Edit with notepad or VSCode
notepad .env
# or
code .env
```

**Add your credentials:**
```
SPOTIFY_CLIENT_ID=your_actual_client_id_here
SPOTIFY_CLIENT_SECRET=your_actual_client_secret_here
SPOTIFY_RATE_LIMIT=100
```

### Step 3: Test API Connection

**Mac:**
```bash
python src/utils/spotify_api.py
```

**Windows:**
```powershell
python src\utils\spotify_api.py
```

You should see audio features for a test track!

---

## Using Jupyter Notebook

### Start Jupyter

```bash
# Make sure aidj environment is active
conda activate aidj

# Start Jupyter
jupyter notebook
```

Browser opens automatically → Navigate to `notebooks/ai_dj_main.ipynb`

### Select Correct Kernel

In Jupyter:
1. Click **Kernel** → **Change Kernel**
2. Select **"Python 3 (aidj)"**

If kernel not found:
```bash
# Install and register kernel
python -m ipykernel install --user --name aidj --display-name "Python 3 (aidj)"
```

---

## Daily Workflow

### On Mac (Development):

```bash
# 1. Activate environment
conda activate aidj

# 2. Start Jupyter or run scripts
jupyter notebook
# or
python your_script.py

# 3. Commit and push changes
git add .
git commit -m "Your message"
git push
```

### On Windows (Testing/Running):

```powershell
# 1. Pull latest changes
git pull

# 2. Activate environment
conda activate aidj

# 3. Run code
jupyter notebook
# or
python your_script.py
```

---

## Managing the Conda Environment

### Activate (every time you work on project)
```bash
conda activate aidj
```

### Deactivate
```bash
conda deactivate
```

### Update environment from file
```bash
conda env update -f environment.yml --prune
```

### List all conda environments
```bash
conda env list
```

### Remove environment (start over)
```bash
conda deactivate
conda env remove -n aidj -y
```

### Export environment (share with others)
```bash
conda env export > environment.yml
```

---

## Platform-Specific Notes

### Mac

- Use `/` for paths
- Use `bash` or `zsh` shell
- Python command: `python` or `python3`

### Windows

- Use `\` for paths (or `/` works too in PowerShell)
- Use PowerShell (not CMD)
- Python command: `python`

### Cross-Platform Commands

These work on both:
```bash
conda activate aidj          # Activate environment
jupyter notebook             # Start Jupyter
python src/utils/config.py   # Run Python (paths auto-converted)
```

---

## Troubleshooting

### "conda: command not found"
**Solution:** Conda not installed or not in PATH.
- Restart terminal after installation
- Or run: `source ~/.bashrc` (Mac) or restart PowerShell (Windows)

### LightFM installation fails
**Solution:** Make sure you're using conda-forge:
```bash
conda install -c conda-forge lightfm -y
```

### Jupyter kernel not found
**Solution:** Register the kernel:
```bash
python -m ipykernel install --user --name aidj
```

### "Module not found" errors
**Solution:** Make sure environment is activated:
```bash
conda activate aidj
# Check which Python you're using:
which python    # Mac
where python    # Windows
```

### Git issues (Windows)
**Solution:** Install Git for Windows:
https://git-scm.com/download/win

---

## Next Steps

Once setup is complete:

1. ✅ Download Spotify dataset (~5GB)
2. ✅ Set up Spotify API credentials
3. ✅ Open `notebooks/ai_dj_main.ipynb`
4. ✅ Start with Section 2: Data Loading and EDA
5. ✅ Follow the project plan in `AI_DJ_Project_Plan.md`

---

## Quick Reference

| Task | Mac | Windows |
|------|-----|---------|
| Activate env | `conda activate aidj` | `conda activate aidj` |
| Start Jupyter | `jupyter notebook` | `jupyter notebook` |
| Run script | `python src/utils/config.py` | `python src\utils\config.py` |
| Git pull | `git pull` | `git pull` |
| Git push | `git add . && git commit -m "msg" && git push` | `git add . && git commit -m "msg" && git push` |

Good luck! 🎵🤖
