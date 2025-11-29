# AI DJ Project - Setup Guide

This guide will walk you through setting up the AI DJ project from scratch.

## Step-by-Step Setup

### Step 1: Install Python Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Spotify Million Playlist Dataset

1. **Create AICrowd Account:**
   - Go to [AICrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
   - Sign up or log in

2. **Download Dataset:**
   - Click "Download Dataset" on the challenge page
   - You'll get a file like `spotify_million_playlist_dataset.zip`
   - This is a large file (~5GB), so download may take a while

3. **Extract to Project:**
   ```bash
   # Extract to data/raw/
   unzip spotify_million_playlist_dataset.zip -d data/raw/
   ```

   The data structure should look like:
   ```
   data/raw/
   ├── data/
   │   ├── mpd.slice.0-999.json
   │   ├── mpd.slice.1000-1999.json
   │   ├── ...
   │   └── mpd.slice.999000-999999.json
   └── README.md
   ```

### Step 3: Set Up Spotify API Credentials

The Spotify API is needed to fetch audio features (BPM, key, energy, etc.) for tracks.

1. **Create Spotify Developer Account:**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Log in with your Spotify account (or create one)

2. **Create an App:**
   - Click "Create an App"
   - App Name: "AI DJ Project" (or anything you like)
   - App Description: "Academic project for playlist generation"
   - Agree to terms and click "Create"

3. **Get Credentials:**
   - Click on your newly created app
   - Click "Settings"
   - Copy your **Client ID** and **Client Secret**

4. **Configure Project:**
   ```bash
   # Copy the example env file
   cp .env.example .env

   # Edit .env and add your credentials
   # Use your favorite text editor (nano, vim, VSCode, etc.)
   nano .env
   ```

   Your `.env` file should look like:
   ```
   SPOTIFY_CLIENT_ID=your_actual_client_id_here
   SPOTIFY_CLIENT_SECRET=your_actual_client_secret_here
   SPOTIFY_RATE_LIMIT=100
   ```

### Step 4: Verify Setup

Test that everything is working:

```bash
# Test configuration
python src/utils/config.py
```

Expected output:
```
AI DJ Configuration
==================================================
Project Root: /path/to/aidj
Data Directory: /path/to/aidj/data
Outputs Directory: /path/to/aidj/outputs

Spotify API:
  Client ID: ✓ Set
  Client Secret: ✓ Set

Validation: ✓ PASSED
```

### Step 5: Test Spotify API

```bash
# Test API connection
python src/utils/spotify_api.py
```

This should fetch audio features for a test track.

### Step 6: Launch Jupyter Notebook

```bash
jupyter notebook notebooks/ai_dj_main.ipynb
```

Your browser should open with the main project notebook.

---

## Project Workflow

Once setup is complete, follow this workflow:

### Phase 1: Data Processing (Week 1)
1. Load raw playlist JSON files
2. Sample 100K playlists with 5-50 tracks
3. Extract unique track URIs
4. Fetch audio features from Spotify API (cached)
5. Create train/val/test splits

### Phase 2: EDA (Week 1)
1. Basic statistics
2. BPM transition analysis
3. Key transition heatmap
4. Energy flow visualization
5. Cold start analysis

### Phase 3: Baseline Models (Week 2)
1. Random baseline
2. Popularity baseline
3. First-order Markov Chain
4. Mean/Linear regression for transitions

### Phase 4: Primary Models (Weeks 3-4)
1. Implement FPMC
2. Implement XGBoost transition model
3. Hyperparameter tuning
4. Evaluation

### Phase 5: Hybrid System (Week 5)
1. Combine FPMC + XGBoost
2. Optimize mixing weights
3. End-to-end evaluation

### Phase 6: Audio Demo (Week 5-6)
1. Install Spleeter
2. Implement crossfading
3. Generate demo playlists

### Phase 7: Presentation (Week 6-7)
1. Create slides
2. Record 20-minute video
3. Export notebook as HTML
4. Submit

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named X"
**Solution:** Make sure you activated the virtual environment and installed requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Spotify API authentication failed"
**Solution:** Check your `.env` file:
- Make sure credentials are correct (no extra spaces)
- Verify your app is active in Spotify Developer Dashboard

### Issue: "Dataset files not found"
**Solution:** Make sure you extracted the dataset to `data/raw/` directory:
```bash
ls data/raw/data/  # Should show mpd.slice.*.json files
```

### Issue: Spleeter installation fails
**Solution:** Spleeter requires TensorFlow 2.12. Try:
```bash
pip install tensorflow==2.12.0
pip install spleeter
```

### Issue: Jupyter kernel crashes
**Solution:** You may need more memory. Try:
- Processing fewer playlists
- Using batch processing
- Clearing variables: `%reset -f`

---

## Directory Reference

```
aidj/
├── data/
│   ├── raw/              # Downloaded Spotify dataset (not in git)
│   ├── processed/        # Sampled and cleaned data
│   ├── features/         # Extracted audio features
│   └── cache/            # Spotify API cache (speeds up reruns)
├── notebooks/
│   └── ai_dj_main.ipynb  # Main submission notebook
├── src/
│   ├── models/           # Model implementations (FPMC, XGBoost, etc.)
│   ├── utils/            # Helper functions (config, API, etc.)
│   └── evaluation/       # Evaluation metrics
├── outputs/
│   ├── figures/          # Plots for presentation
│   ├── audio/            # Generated playlist audio
│   └── results/          # Evaluation results (JSON/CSV)
├── requirements.txt      # Python dependencies
├── .env                  # Your credentials (NOT in git)
├── .env.example          # Template for .env
├── .gitignore            # Git ignore rules
├── README.md             # Project overview
└── SETUP_GUIDE.md        # This file
```

---

## Next Steps

1. **Download the dataset** (this is the longest step)
2. **Set up Spotify API** credentials
3. **Open the main notebook** and start with Section 2 (EDA)
4. **Follow the project plan** in `AI_DJ_Project_Plan.md`

Good luck with your project! 🎵🤖
