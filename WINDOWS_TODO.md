# Windows Setup Checklist

## ✅ What to Do on Your Windows Machine

### Step 1: Pull Latest Changes

```powershell
cd "C:\vscode workspace\aidj\aidj"
git pull
```

### Step 2: Remove Old venv

```powershell
# Deactivate if still active
deactivate

# Delete venv folder (or delete manually in File Explorer)
Remove-Item -Recurse -Force venv
```

### Step 3: Create Conda Environment

```powershell
# Create environment from file
conda env create -f environment.yml

# This will:
# - Create "aidj" conda environment
# - Install Python 3.11
# - Install LightFM from conda-forge (no build issues!)
# - Install all other dependencies
```

### Step 4: Activate Environment

```powershell
conda activate aidj
```

You should see `(aidj)` in your prompt.

### Step 5: Verify Installation

```powershell
# Test configuration
python src\utils\config.py
```

**Expected output:**
```
AI DJ Configuration
==================================================
...
Spotify API:
  Client ID: ✗ Not Set    ← This is OK for now
  Client Secret: ✗ Not Set ← This is OK for now
```

### Step 6: Set Up Spotify API (If Not Done Yet)

Only if you haven't created `.env` file:

```powershell
# Copy example
copy .env.example .env

# Edit with notepad or VSCode
notepad .env
```

Add your credentials:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### Step 7: Test API

```powershell
python src\utils\spotify_api.py
```

Should show audio features for a test track.

### Step 8: Start Jupyter

```powershell
jupyter notebook
```

Navigate to `notebooks/ai_dj_main.ipynb`

---

## ✅ You're Done!

Your environment is now set up and identical to Mac.

---

## Daily Workflow

Every time you work on the project:

```powershell
# 1. Pull latest changes from Mac
git pull

# 2. Activate environment
conda activate aidj

# 3. Start working
jupyter notebook
# or
python your_script.py
```

---

## Troubleshooting

### conda not found
- Make sure Anaconda is installed
- Restart PowerShell
- Or use Anaconda Prompt

### Environment creation fails
Try manual installation (see SETUP.md)

### LightFM still won't install
We have a backup plan - implement FPMC from scratch (actually better for the assignment!)

---

## Need Help?

See detailed guide: [SETUP.md](SETUP.md)
