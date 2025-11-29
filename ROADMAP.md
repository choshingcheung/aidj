# Project Roadmap & Progress Tracker

Track implementation progress for the AI DJ project. Check off items as you complete them.

---

## Phase 1: Setup & Data Acquisition
**Timeline**: Days 1-2

- [x] Set up conda environment
- [x] Configure Spotify API credentials
- [ ] Download Spotify Million Playlist Dataset (~5GB)
- [ ] Verify dataset structure (1000 JSON slice files)
- [ ] Test data loading with sample file

---

## Phase 2: Data Preprocessing & Feature Engineering
**Timeline**: Days 3-4

### Data Sampling & Cleaning
- [ ] Load and sample 100K playlists (5-50 tracks each)
- [ ] Remove duplicates and validate track URIs
- [ ] Filter rare songs (< 5 appearances)
- [ ] Create train/val/test splits (70/15/15)

### Feature Extraction
- [ ] Fetch audio features from Spotify API for all unique tracks
- [ ] Implement caching to avoid redundant API calls
- [ ] Handle rate limits and missing data
- [ ] Save processed datasets to `data/processed/`

### Transition Features
- [ ] Compute BPM differences
- [ ] Calculate key distance (circle of fifths)
- [ ] Compute energy, valence, danceability deltas
- [ ] Create binary features (mode match, harmonic compatibility)
- [ ] Generate smoothness scores for ground truth transitions

---

## Phase 3: Exploratory Data Analysis
**Timeline**: Days 5-7

### Required Analyses (5+ visualizations)
- [ ] **Analysis 1**: Basic statistics (playlist length, track frequency distributions)
- [ ] **Analysis 2**: BPM transition histogram (show smooth transitions preference)
- [ ] **Analysis 3**: Key transition heatmap (circle of fifths patterns)
- [ ] **Analysis 4**: Energy flow over playlist position (typical arcs)
- [ ] **Analysis 5**: Cold start analysis (rare song frequency distribution)
- [ ] **Analysis 6+**: Additional insights (optional)

### Insights & Documentation
- [ ] Write analysis narratives for each visualization
- [ ] Connect observations to model design choices
- [ ] Statistical tests for significance
- [ ] Export plots to `outputs/figures/`

---

## Phase 4: Baseline Models
**Timeline**: Days 8-9

### Sequential Prediction Baselines
- [ ] **Random Baseline**: Uniform random selection
- [ ] **Popularity Baseline**: Top-N most popular tracks
- [ ] **First-Order Markov Chain**: Build transition matrix from co-occurrences

### Transition Quality Baselines
- [ ] **Mean Baseline**: Predict average smoothness
- [ ] **Linear Regression**: Train on 13 transition features

### Evaluation Framework
- [ ] Implement Hit@K metric (K=5, 10, 20)
- [ ] Implement AUC metric
- [ ] Implement MSE, MAE, R² metrics
- [ ] Create reusable evaluation functions
- [ ] Baseline results table

---

## Phase 5: FPMC Implementation
**Timeline**: Days 10-12

### Model Implementation
- [ ] Implement FPMC class with BPR loss
- [ ] User (playlist) embeddings
- [ ] Item (song) embeddings
- [ ] Sequential transition embeddings
- [ ] Mini-batch SGD training loop

### Training & Tuning
- [ ] Prepare training data (user, prev_item, next_item) tuples
- [ ] Hyperparameter grid search:
  - [ ] Embedding dimensions: {32, 64, 128}
  - [ ] Learning rate: {0.001, 0.01, 0.1}
  - [ ] Regularization: {0.0001, 0.001, 0.01}
- [ ] Train on train set, tune on validation set
- [ ] Save best model checkpoint

### Evaluation
- [ ] Compute Hit@K on test set
- [ ] Compute AUC on test set
- [ ] Compare to baselines
- [ ] Statistical significance tests

---

## Phase 6: XGBoost Transition Model
**Timeline**: Days 13-14

### Model Training
- [ ] Prepare transition feature matrix (13 features)
- [ ] Prepare target smoothness scores
- [ ] Train XGBoost regressor

### Hyperparameter Tuning
- [ ] n_estimators: {50, 100, 200}
- [ ] max_depth: {3, 5, 7}
- [ ] learning_rate: {0.01, 0.1, 0.3}

### Analysis
- [ ] Compute MSE, MAE, R² on test set
- [ ] Extract and visualize feature importance
- [ ] Validate learned patterns (e.g., BPM matters most)
- [ ] Compare to baseline regression

---

## Phase 7: Hybrid System
**Timeline**: Days 15-16

### Integration
- [ ] Implement hybrid scoring function
- [ ] FPMC generates top-K candidates (K=50-100)
- [ ] XGBoost scores each candidate for transition quality
- [ ] Combined score: α × P_seq + β × Q_trans

### Optimization
- [ ] Grid search for α: {0.1, 0.3, 0.5, 0.7, 0.9}
- [ ] Evaluate on validation set
- [ ] Select best α value

### Final Evaluation
- [ ] Evaluate hybrid system on test set
- [ ] Hit@K performance
- [ ] Average playlist smoothness
- [ ] Comparison table: Random → Popularity → Markov → FPMC → Hybrid
- [ ] Statistical significance tests

---

## Phase 8: Demo & Audio Generation
**Timeline**: Days 17-18

### Spleeter Setup
- [ ] Install Spleeter (mark as pretrained in documentation)
- [ ] Test audio source separation on sample track

### Demo Playlist Generation
- [ ] Generate 3 example playlists:
  - [ ] **Playlist 1**: "Morning Workout" (energy ramp-up, high BPM)
  - [ ] **Playlist 2**: "Evening Chill" (low energy, slow BPM)
  - [ ] **Playlist 3**: Failure case (acknowledge limitations)

### Audio Crossfading
- [ ] Separate tracks into stems (vocals, drums, bass, other)
- [ ] Implement intelligent crossfade based on transition quality
- [ ] Reconstruct mixed audio output
- [ ] Save to `outputs/audio/`

### Visualizations
- [ ] BPM/energy curves over playlist position
- [ ] Transition quality heatmap for demo playlists
- [ ] Save figures for presentation

---

## Phase 9: Related Work & Literature
**Timeline**: Day 19

### Literature Review
- [ ] Read and cite FPMC paper (Rendle et al., 2010)
- [ ] Read and cite compatibility paper (McAuley et al., 2015)
- [ ] Read Spleeter paper (Hennequin et al., 2020) - mark as pretrained
- [ ] Review music recommendation surveys
- [ ] Find 2-3 additional relevant papers

### Positioning
- [ ] Write "Related Work" section in notebook
- [ ] Explain how prior work differs from our approach
- [ ] Clearly state our novel contribution
- [ ] Compare results to reported benchmarks (if available)

---

## Phase 10: Presentation & Submission
**Timeline**: Days 20-21

### Notebook Finalization
- [ ] Ensure all cells run without errors
- [ ] Add markdown documentation to all sections
- [ ] Remove debug output and print statements
- [ ] Verify plots are high quality
- [ ] Export as HTML: `workbook.html`
- [ ] Test that HTML opens in browser

### Video Presentation (18-20 minutes)
- [ ] **Section 1**: Predictive tasks & evaluation (3-4 min)
- [ ] **Section 2**: EDA & data insights (4-5 min)
- [ ] **Section 3**: Modeling & architecture (5-6 min)
- [ ] **Section 4**: Results & comparisons (5-6 min)
- [ ] **Section 5**: Related work (2-3 min)
- [ ] Record video (Zoom, OBS, etc.)
- [ ] Upload to Google Drive or YouTube
- [ ] Test video link is accessible

### Submission Files
- [ ] `workbook.html` (exported notebook)
- [ ] `video_url.txt` (single line with video link)
- [ ] Verify files with autograder script (if provided)
- [ ] Submit to Gradescope by deadline

### Final Checks
- [ ] Presentation matches code
- [ ] No auto-generated content
- [ ] Spleeter clearly marked as pretrained
- [ ] All models from course content included
- [ ] Video length: 18-22 minutes
- [ ] Video quality: watchable, clear audio

---

## Grading Rubric Reference

Each of 5 sections worth 5 marks:

- **0**: Not covered
- **1**: Covered but has errors or unclear
- **2**: Superficial, missing key elements
- **3**: Minimally acceptable, essential elements present
- **4**: Feature-complete, results make sense (target grade)
- **5**: Goes above and beyond

**Peer grading**: +4 marks (due 1 week after submission)

**Total**: 29 marks

---

## Progress Summary

Track your overall progress:

- [ ] Phase 1: Setup ✓
- [ ] Phase 2: Data Processing
- [ ] Phase 3: EDA
- [ ] Phase 4: Baselines
- [ ] Phase 5: FPMC
- [ ] Phase 6: XGBoost
- [ ] Phase 7: Hybrid System
- [ ] Phase 8: Demo & Audio
- [ ] Phase 9: Related Work
- [ ] Phase 10: Presentation & Submission

---

**Last Updated**: [Add date when you update progress]

**Current Phase**: Phase 1 ✓ (Setup Complete)

**Next Milestone**: Download dataset & begin preprocessing
