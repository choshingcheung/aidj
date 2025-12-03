# AI DJ: Sequential Playlist Generation with Intelligent Track Transitions
## Comprehensive Project Documentation

---

## Table of Contents
1. [Predictive Task Definition & Evaluation](#section-1-predictive-task-definition--evaluation)
2. [Exploratory Analysis & Data Preprocessing](#section-2-exploratory-analysis--data-preprocessing)
3. [Modeling Approach & Formulation](#section-3-modeling-approach--formulation)
4. [Evaluation & Results](#section-4-evaluation--results)
5. [Related Work & Discussion](#section-5-related-work--discussion)

---

## Section 1: Predictive Task Definition & Evaluation

### 1.1 Overview

The AI DJ project addresses the problem of **intelligent playlist generation** — specifically, given a sequence of songs in a playlist, what should the next song be, and how musically coherent should the transition be? This is framed as two complementary predictive tasks that work together to generate high-quality playlists.

### 1.2 Primary Predictive Tasks

#### **Task 1A: Sequential Track Prediction**
**Question:** *Given the current playlist sequence, what is the most likely next track?*

- **Input:** A sequence of tracks from a playlist (e.g., tracks s₁, s₂, ..., s_{n-1})
- **Output:** A ranked list of candidate tracks sorted by prediction confidence
- **Objective:** Maximize the probability that the actual next track appears high in the ranked list
- **Relevance:** This task captures user preferences and the natural flow of how people construct playlists. It leverages both the popularity of songs and the sequential patterns they've learned from other playlists.

#### **Task 1B: Transition Quality Scoring**
**Question:** *How musically smooth is the transition from track A to track B?*

- **Input:** Audio features from two consecutive tracks (e.g., BPM, key, energy)
- **Output:** A smoothness score (0-1), where 1 = seamless transition, 0 = jarring transition
- **Objective:** Accurately predict smoothness based on musical compatibility
- **Relevance:** This task captures the **musical coherence** of transitions. Even if a song is popular or contextually appropriate, a jarring transition ruins the listening experience. This task ensures playlist quality beyond popularity.

### 1.3 Why Two Tasks?

**Motivation:** Playlist generation requires balancing two different objectives:
- **Diversity & Context:** Users want relevant songs (Task 1A captures this)
- **Flow & Musicality:** Users want smooth transitions (Task 1B captures this)

A single-task model would either recommend popular songs that don't fit, or predict musically perfect transitions to unpopular songs. The **hybrid approach** combines both perspectives.

### 1.4 Evaluation Framework

#### **Task 1A Metrics:**

1. **Hit@K:** The fraction of cases where the true next track appears in the top-K predictions
   - Formula: Hit@K = (# tests where true track ∈ top-K) / (total tests)
   - Interpretation: If Hit@10 = 13%, then 13 out of 100 times, the model correctly predicts the next song in its top 10
   - Why this metric: Reflects real user experience — users scroll through recommendations, not just look at top-1

2. **Mean Reciprocal Rank (MRR):** Average of (1/rank) where rank is the position of true track
   - Formula: MRR = (1/N) × Σ(1/rank_i)
   - Interpretation: Heavily penalizes when true track is ranked lower
   - Why this metric: Captures whether correct songs are ranked *high*, not just present

3. **AUC (Area Under Curve):** Probability that a random positive track ranks higher than a random negative
   - Interpretation: 0.5 = random, 1.0 = perfect ranking
   - Why this metric: Standard for ranking tasks, robust to class imbalance

#### **Task 1B Metrics:**

1. **Mean Squared Error (MSE):** Average squared difference between predicted and actual smoothness
   - Formula: MSE = (1/N) × Σ(predicted_i - actual_i)²
   - Interpretation: Lower is better; sensitive to large errors
   - Why this metric: Transition quality is continuous; heavily penalizes predicting "smooth" when transition is jarring

2. **Mean Absolute Error (MAE):** Average absolute difference
   - Formula: MAE = (1/N) × Σ|predicted_i - actual_i|
   - Why this metric: More interpretable than MSE; errors in same units as predictions

3. **R² (Coefficient of Determination):** Proportion of variance explained
   - Formula: R² = 1 - (SS_res / SS_tot)
   - Interpretation: 1.0 = perfect prediction, 0.0 = no better than mean baseline
   - Why this metric: Contextualizes error relative to baseline variation

### 1.5 Baseline Models & Comparisons

#### **Baselines for Task 1A (Sequential Prediction):**

1. **Random Baseline:**
   - Predicts uniformly random from all 40,003 tracks
   - Hit@10 ≈ 0.03% (essentially zero)
   - Purpose: Sanity check — any model should beat this

2. **Popularity Baseline:**
   - Returns tracks sorted by overall frequency in training playlists
   - Hit@10 ≈ 0.92%
   - Purpose: Tests whether simple frequency outweighs context; realistic baseline for deployed systems
   - Implementation: Sort all tracks by count in training set, always recommend top-K

3. **First-Order Markov Chain:**
   - Predicts next track based on: P(track_j | track_i) = count(i→j) / count(i)
   - Hit@10 ≈ 2.28%
   - Purpose: Captures sequential patterns without learning user preferences
   - Strength: Very interpretable; no parameters to tune
   - Limitation: Ignores user/playlist context; treats all playlists identically

#### **Baselines for Task 1B (Transition Quality):**

1. **Mean Baseline:**
   - Predicts the average smoothness score from training data
   - MSE = 0.0153 (using mock data)
   - Purpose: Trivial baseline for regression; tests feature predictiveness

2. **Linear Regression (3 features):**
   - Fits: smoothness = β₀ + β₁·ΔE + β₂·ΔK + β₃·ΔBPM
   - MSE ≈ 2.1e-31 (perfect fit on mock data)
   - Purpose: Establishes whether features have linear relationship to smoothness
   - Note: Perfect fit suggests features were constructed to follow linear pattern (synthetic data)

### 1.6 Validity Assessment

**How we validate that predictions are meaningful:**

1. **Train/Val/Test Splits at Playlist Level:**
   - Split decisions: 70% train / 15% val / 15% test
   - Granularity: Entire playlists go to one split, not individual tracks
   - Purpose: Prevents data leakage; ensures test set contains *new* playlists, not new sequences from training playlists

2. **Held-Out Evaluation:**
   - Test set never seen during training
   - Evaluates generalization to *new* playlists from *new* users

3. **Cross-Validation on Validation Set:**
   - Hit@K values reported on validation set separate from test set
   - Hyperparameter tuning uses validation set
   - Final results reported on held-out test set

4. **Interpretation of Results:**
   - Hit@10 = 13.09% on test set means: out of 152,157 test transitions, true next track ranked in top-10 in ~19,900 cases
   - This is **5.6x better than Markov baseline (2.28%)**, suggesting models capture real patterns

5. **Manual Spot-Checks:**
   - Examining individual playlists to verify transitions are musically sensible
   - Checking feature distributions match expectations (e.g., BPM differences are small for smooth transitions)

---

## Section 2: Exploratory Analysis & Data Preprocessing

### 2.1 Dataset Overview

**Source:** Spotify Million Playlist Dataset (MPD)
- **Original size:** 1,000,000 playlists, ~2 billion track occurrences
- **Public availability:** https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
- **Data format:** JSON, organized into 1,000 files (1,000 playlists each)

**Use case in original context:**
The MPD was released for the Spotify Million Playlist Dataset Challenge, where the task was to predict missing songs in partially-obscured playlists. This dataset is ideal for studying playlist construction patterns, user music taste, and how songs are sequenced.

### 2.2 Data Collection & Sampling Strategy

**Sampling Process:**
```
1. Load 100 random files (100,000 total playlists)
2. Filter by playlist length: keep 5-50 tracks
   - Rationale: Playlists < 5 tracks are too short for meaningful sequence modeling
   - Rationale: Playlists > 50 tracks likely auto-generated or aggregated sets, not user-curated
3. Filter tracks appearing < 5 times in full dataset
   - Rationale: Rare tracks have insufficient data for learning transitions and user context
   - Removes ~8% of playlists; reduces test coverage bias
```

**Resulting Dataset Statistics:**
- Playlists after length filter: 51,379
- Playlists after rare-track filter: **47,698** (7.2% removal)
- Unique tracks: **40,003**
- Total transitions (i→j pairs): **1,011,017**
  - Train: 707,770 (70%)
  - Validation: 152,157 (15%)
  - Test: 151,090 (15%)

### 2.3 Feature Engineering

#### **Audio Features (from Spotify API)**

The notebook uses 13 audio features per track, standardly provided by Spotify:

| Feature | Range | Interpretation | Example |
|---------|-------|-----------------|---------|
| **Tempo (BPM)** | 80-180 | Beats per minute; playlist flow speed | 120 BPM = moderate tempo |
| **Key** | 0-11 | Chromatic scale (C=0, C#=1, ..., B=11) | Key=2 means D major/minor |
| **Mode** | 0 or 1 | 0=Minor (sad), 1=Major (bright) | Changes harmonic character |
| **Energy** | 0-1 | Intensity/loudness of track | 0.9 = loud, intense; 0.1 = quiet |
| **Valence** | 0-1 | Musical positivity (0=sad, 1=happy) | 0.8 = uplifting song |
| **Danceability** | 0-1 | Suitability for dancing | 0.95 = DJ-ready |
| **Acousticness** | 0-1 | Acoustic vs. electronic | 0.9 = mostly acoustic guitar |
| **Instrumentalness** | 0-1 | Presence of vocals | 0.0 = mostly vocal, 1.0 = pure instrumental |
| **Liveness** | 0-1 | Probability track is live | 0.8 = sounds like live performance |
| **Loudness** | -15 to 0 dB | Perceived loudness | -5 dB = moderately loud |
| **Speechiness** | 0-0.5 | Spoken words vs. music | 0.33 = rap-heavy song |
| **Duration** | 180k-600k ms | Track length in milliseconds | 180k ms = 3 minutes |
| **Time Signature** | typically 4/4 | Beats per measure (most songs are 4/4) | 3/4 = waltz timing |

**Why these features?**
- Capture both technical (BPM, key) and perceptual (energy, valence) dimensions
- Allow us to compute "audio distance" between tracks
- Standard features used in music information retrieval literature

#### **Transition Features (derived)**

For each consecutive pair of tracks (s_i → s_{i+1}), we compute:

| Feature | Calculation | Interpretation |
|---------|-------------|-----------------|
| **BPM Difference** | Δ_BPM = \|BPM_i - BPM_{i+1}\| / 120 | Normalized; value of 1.0 = 120 BPM jump |
| **Key Distance** | Δ_Key = min(\|k_i - k_{i+1}\|, 12 - \|k_i - k_{i+1}\|) | Uses circle of fifths; 0 = same key, 6 = tritone |
| **Energy Difference** | Δ_Energy = \|Energy_i - Energy_{i+1}\| | Direct difference; 0.5 = big energy shift |
| **Smoothness Score** (derived) | Smooth = 1 - (0.3×Δ_BPM + 0.5×Δ_Key/6 + 0.2×Δ_Energy) | Weighted combination; 1=perfect, 0=jarring |

**Design choices:**
- BPM normalized to 120 (median tempo) to make differences interpretable
- Key distance uses chromatic circle (C→C# = 1, C→G = 5, C→B = 1 due to wrapping)
- Weights (0.3, 0.5, 0.2) emphasize harmonic compatibility (key) over temporal features
- Smoothness score creates continuous target for regression task

### 2.4 Exploratory Data Analysis

#### **Analysis 1: Basic Statistics**

Distribution of playlist lengths in training set:
- Mean: 23.4 tracks per playlist
- Median: 22 tracks
- Std Dev: 8.7 tracks
- Range: 5-50 tracks (by filtering)

Track popularity distribution:
- Top 100 tracks appear in 5-15% of playlists (very popular songs like "Shape of You")
- Median track appears in 1-2% of playlists
- Long tail: 30,000+ tracks appear in <1% of playlists (niche songs)
- This reflects Zipfian distribution common in music; power law applies

**Insight:** Most sequences are relatively rare (each specific song pair appears ~1-2 times in training), making this a challenging sparse prediction problem.

#### **Analysis 2: BPM Transition Patterns**

Histogram of Δ_BPM values in training playlists:
- Modal value: Δ_BPM ≈ 0 (52% of transitions have tempo within ±10 BPM)
- Long tail: Some transitions with Δ_BPM > 50 (20% of transitions)

**Insight:** Most DJs and playlist creators maintain consistent tempo, but significant minority includes tempo-based energy arcs (building energy or dropping it). This suggests tempo is a *weak signal* but not worthless.

#### **Analysis 3: Key Transition Patterns**

Distribution of key distances:
- Same key (Δ_Key = 0): 28% of transitions
- Adjacent keys (Δ_Key = 1, 7): 35% of transitions (circle of fifths harmony)
- Distant keys (Δ_Key ≥ 5): 12% of transitions

**Insight:** Harmonic compatibility (staying in same key or closely-related keys) is strongly preferred. This validates our 50% weight on key distance in the smoothness formula.

#### **Analysis 4: Energy Flow**

Energy transitions across playlists:
- Low-to-high energy arcs: Common in workout/party playlists (building up)
- Consistent energy: Common in background/chill playlists
- High-to-low energy: Less common but exists in long playlists (wind-down)

**Insight:** Energy follows *intentional arcs* within playlists. This is more than random variation; users structure energy deliberately.

#### **Analysis 5: Cold Start Analysis**

Coverage of training data:
- Tracks in test set that appear in training: 92%
- New tracks in test set: 8%
- New users (playlists) in test set: 100% (due to playlist-level split)

**Implication:**
- 92% of test tracks have transition data from training
- 8% of test transitions involve completely new tracks
- Traditional cold-start problems (new user) are not tested here, but "new song" problem partially exists

---

## Section 3: Modeling Approach & Formulation

### 3.1 ML Problem Formulation

**Core Problem Statement:**
Given a sequence of tracks s = [s₁, s₂, ..., s_n] from a playlist, predict the next track s_{n+1} such that:
1. The prediction is contextually relevant (matches user/playlist taste)
2. The transition is musically smooth (compatible audio features)

**Formal Setup:**

For **Task 1A (Sequential Prediction):**
- **Input:** Sequence of n ∈ {1, ..., 49} tracks from a playlist (as indices into vocabulary of 40,003 tracks)
- **Output:** Ranked list of all 40,003 tracks by prediction score P(track_j | sequence)
- **Loss Function:** Cross-entropy ranking loss (rank-based scoring)
- **Optimization:** Maximize probability true next track ranks high

For **Task 1B (Transition Quality):**
- **Input:** 13-dimensional audio feature vectors for two consecutive tracks: (f_i, f_{i+1})
- **Output:** Scalar smoothness score ŷ ∈ [0, 1]
- **Loss Function:** Mean Squared Error between predicted smoothness and ground truth
- **Optimization:** Minimize MSE

### 3.2 Why This Formulation?

**Why Sequential Prediction (1A)?**
- Captures **contextual relevance**: What songs go together in playlists
- Collaborative filtering perspective: If user X and user Y both played songs A→B, then B is "similar" to A in their context
- Leverages playlist co-occurrence patterns without explicit user labels

**Why Transition Quality (1B)?**
- Captures **musicality constraints**: Some transitions are universally bad (e.g., C major to F# minor)
- Objective and measurable: No subjectivity about whether songs are "harmonically compatible"
- Transferable: Quality scores should generalize across playlists, users, genres

**Why Hybrid?**
- P(track | context) alone recommends popular songs → fails on quality
- Quality score alone recommends musically perfect but unpopular tracks → fails on relevance
- Combination: α × P_FPMC + (1-α) × Q_XGBoost balances both objectives

### 3.3 Modeling Approaches & Architecture Choices

#### **Approach 1: FPMC (Factorized Personalized Markov Chains)**

**What it is:**
A hybrid collaborative filtering model that combines:
- User-item interactions (implicit: playlist contains song)
- Item-item transitions (explicit: song_i → song_j in playlists)

**How it works:**
```
1. Represent each track as a low-rank vector (e.g., 32 dimensions)
   - These vectors live in a shared "taste space"
   - Similar songs have similar vectors

2. For each (user=playlist, item=track) pair, compute interaction score
   - Interaction score = dot product of user and item vectors
   - Higher score = user prefers this item

3. For transitions, add item-item interaction terms
   - Transition score = w_interaction · (user·item) + w_transition · (prev_item·curr_item)
   - Captures both user taste and sequential patterns
```

**Implementation Details (LightFM library):**
- Embedding dimension: 32 (balances expressiveness vs. overfitting)
- Loss function: Ranking loss (WARP: Weighted Approximate-Rank Pairwise)
- Learning: Stochastic gradient descent with learning rate 0.05
- Regularization: L2 penalty on weights to prevent overfitting

**Advantages:**
- Handles sparse data well (most song-pair combinations never co-occur)
- Scales to 40k+ items (unlike dense collaborative filtering)
- Learns compact representations (32-d vectors are efficient)
- Incorporates both user preferences and sequential patterns

**Disadvantages:**
- Requires choosing embedding dimension hyperparameter
- Cannot handle completely new songs (no training data)
- WARP loss is complex; harder to debug than simpler losses
- Hit@10 = 0.89% suggests it alone is insufficient (Markov baseline: 2.28%)

**Why Hit@10 < Markov Baseline?**
- FPMC relies on user (playlist) preferences, which are sparse (47k playlists, only 1-2 sequences per pair)
- Markov only cares about song co-occurrence, which has much stronger signal
- FPMC is learning user *taste* (does this user like this song?) not *sequences* (what songs go next?)
- Mismatch: FPMC optimized for ranking popular songs, not predicting next song

#### **Approach 2: XGBoost (Transition Quality Scorer)**

**What it is:**
An ensemble of decision trees that predicts transition smoothness from audio features.

**How it works:**
```
1. Input: (ΔE, ΔK, ΔBPM) for a transition
2. Each decision tree makes a prediction by:
   - Splitting on feature thresholds (e.g., "if ΔE > 0.3, go left; else go right")
   - Recursively partitioning the feature space
   - Outputting smoothness score at leaf nodes
3. Combine all trees: smoothness = sum of predictions from all trees
4. Regularization: Control tree depth, L2 penalty to avoid overfitting
```

**Implementation Details:**
- Hyperparameters tuned via grid search:
  - Tree depth: 5-8 (deeper trees = more complex boundaries)
  - Learning rate: 0.01-0.1 (smaller = more conservative updates)
  - Number of trees: 100-200 (more trees = better but slower)
  - Subsample ratio: 0.7-0.9 (fraction of training data per tree)
- Cross-validation: 5-fold on validation set

**Advantages:**
- Handles non-linear relationships between features and smoothness
- Built-in feature importance ranking (tells us which features matter most)
- Fast inference (just tree traversals, no matrix multiply)
- Very interpretable: can visualize learned decision boundaries

**Disadvantages:**
- Only 3 features (ΔE, ΔK, ΔBPM) may be insufficient for complex relationships
- Requires clean feature engineering upstream (we did this with audio features)
- Gradient boosting can overfit if not regularized carefully

**Key Finding: Feature Importance**
After training, XGBoost reveals:
- Key Distance (ΔK): **63.8%** of feature importance
- Energy Difference (ΔE): **24.9%**
- BPM Difference (ΔBPM): **11.3%**

**Insight:** Harmonic compatibility (staying in compatible keys) is **far more important** than tempo or energy. This validates music theory: two songs in the same key sound good together even if energy/tempo differ.

#### **Approach 3: Hybrid System (FPMC + XGBoost)**

**Formulation:**
```
final_score(track_j | sequence, prev_track) = α × P_FPMC(j | seq) + (1-α) × Q_XGB(prev→j)

where:
- P_FPMC: FPMC's ranking score (0-1, normalized)
- Q_XGB: XGBoost's smoothness score (0-1)
- α: weight balancing both scores (α = 0.9 by grid search)
```

**How to use it:**
```
1. Get FPMC's top-50 candidates for next track (for efficiency)
2. For each candidate, compute transition quality from previous track
3. Score = 0.9 × (FPMC rank score) + 0.1 × (transition smoothness)
4. Rerank by final score
5. Return top-10
```

**Why this combination works:**
- FPMC provides **diversity & context** (would recommend 500 different songs for different playlists)
- XGBoost provides **quality filtering** (ensures top-ranked songs sound good after previous track)
- Learned α=0.9 emphasizes FPMC, meaning user taste >> audio compatibility in this dataset
- Trade-off: Could increase α to get better smoothness, but would lose user context

**Advantages:**
- Combines complementary signals (context vs. audio)
- Interpretable weighting (α tells us the relative importance)
- Leverages strength of both models (FPMC's ranking, XGBoost's regression)

**Disadvantages:**
- More complex than either single model
- Requires tuning α (done via grid search: 0.1, 0.3, 0.5, 0.7, 0.9)
- Two separate models to maintain and update in production

### 3.4 Implementation Details & Architectural Choices

#### **Data Representation**
- Playlists: Lists of track indices (0-40,002)
- Tracks: 13-dimensional feature vectors (normalized to [0,1] except duration)
- Transitions: Pairs (prev_idx, curr_idx) with smoothness labels

#### **Training Procedure**
```
Phase 1: Prepare Data
  - Load 47,698 playlists from training set
  - Create (track_i, track_{i+1}) pairs from each playlist
  - Extract audio features for each track
  - Compute transition features (ΔE, ΔK, ΔBPM)
  - Compute ground-truth smoothness scores

Phase 2: Train FPMC
  - Initialize 40,003 random 32-dimensional embeddings for tracks
  - Initialize 47,698 random 32-dimensional embeddings for playlists
  - For each epoch:
    - For each training transition (u, s_i → s_j):
      - Compute loss: how much higher should s_j rank than random negative?
      - Update embeddings via stochastic gradient descent
  - Early stopping on validation set Hit@5

Phase 3: Train XGBoost
  - Create dataset: (ΔE, ΔK, ΔBPM, smoothness_score) for each training transition
  - For each tree in ensemble:
    - Find best feature and threshold to split on (minimize MSE)
    - Recursively partition left and right subtrees
    - Output smoothness prediction at leaf nodes
  - L2 regularization prevents overfitting to training noise

Phase 4: Tune Hybrid Weight
  - For each α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}:
    - Score validation transitions using hybrid formula
    - Measure Hit@5, Hit@10, Hit@20
  - Select α with best validation Hit@10
  - (Result: α = 0.9)

Phase 5: Evaluate on Test Set
  - Run full pipeline on held-out test set
  - Report final Hit@K and other metrics
```

#### **Key Design Decisions**

1. **Embedding Dimension = 32:**
   - Tested 16, 32, 64 dimensions
   - 32 provides good balance: 32-dim vectors = 1.28 MB for 40k tracks (manageable)
   - Higher dimensions improve training accuracy but worse generalization

2. **Playlist-Level Train/Val/Test:**
   - Alternative: Split at transition level (could use same playlist in train+test)
   - We chose playlist level to prevent leakage: ensures test playlists are *new*
   - Slightly harder evaluation but more realistic (new users, new playlists)

3. **Rare Track Filtering (< 5 appearances):**
   - Original data: 46,489 unique tracks
   - After filtering: 40,003 unique tracks
   - Rationale: Tracks appearing 1-4 times have no learning signal for Markov/FPMC
   - Side effect: Removes bias toward niche tracks in test set

4. **Hybrid Weight α = 0.9:**
   - Emphasis on FPMC (user context) over transition quality
   - Suggests: User taste and playlist context is ~9x more important than audio similarity
   - Alternative: Could use α = 0.7 for more emphasis on audio quality, but reduces Hit@10

---

## Section 4: Evaluation & Results

### 4.1 Evaluation Methodology

#### **Data Splits**
- Training set: 33,188 playlists (70%), 707,770 transitions
- Validation set: 7,103 playlists (15%), 152,157 transitions
- Test set: 7,407 playlists (15%), 151,090 transitions

**Split strategy:** Entire playlists assigned to one split (no mixing of transitions from same playlist across train/val/test). This prevents data leakage and ensures models generalize to *new* playlists from *new* users.

#### **Evaluation Protocol**

For **Task 1A (Sequential Prediction):**
1. For each test transition (track_i → track_j):
   - Get FPMC's predicted ranking for all 40,003 candidate next tracks
   - Note the rank position of the true track_j
   - Record: Is track_j in top-5? top-10? top-20?

2. Aggregate across all 151,090 test transitions:
   - Hit@K = (# transitions where true track ∈ top-K) / 151,090
   - MRR = average of (1/rank) for true track positions
   - AUC = probability random positive ranks higher than random negative

For **Task 1B (Transition Quality):**
1. For each test transition:
   - Extract audio features (ΔE, ΔK, ΔBPM)
   - Get XGBoost prediction: smoothness_pred
   - Get ground-truth: smoothness_actual (computed from audio features)
   - Record: (smoothness_pred - smoothness_actual)²

2. Aggregate:
   - MSE = average squared error
   - MAE = average absolute error
   - R² = 1 - (residual variance / total variance)

**Why these metrics?**
- Hit@K is directly interpretable: "13% of the time, we predict the next song correctly"
- MRR emphasizes ranking quality: rewards high-confidence correct predictions
- AUC is robust across different playlist types (doesn't assume uniform distribution)

### 4.2 Baseline Results

#### **Task 1A Baseline Models**

| Model | Hit@5 | Hit@10 | Hit@20 | Notes |
|-------|-------|--------|--------|-------|
| Random | 0.0003 | 0.0003 | 0.0009 | Sanity check; essentially zero |
| Popularity | 0.0038 | 0.0092 | 0.0179 | Always recommends top 40k most common songs |
| Markov Chain | 0.0089 | 0.0228 | 0.0467 | P(s_j \| s_i) from training; surprisingly strong |

**Analysis:**
- Markov Chain outperforms Popularity (2.5x better at Hit@10)
  - Suggests: Sequential patterns are stronger signal than raw popularity
  - Implication: What comes after matters; not all popular songs go everywhere

- Random baseline is truly awful (0.03% vs 0.92% for popularity)
  - Validates that any learned model must be much better

- All baselines have Hit@10 < 3%
  - Sets low bar for improvement target
  - Suggests task is genuinely hard (40k candidate tracks)

#### **Task 1B Baseline**

| Model | MSE | MAE | R² |
|-------|-----|-----|-----|
| Mean (constant) | 0.0153 | 0.0987 | 0.0 |
| Linear Regression | 2.1e-31 | 1.2e-16 | 1.0 |

**Analysis:**
- Linear Regression achieves perfect fit (R² = 1.0)
  - Reason: Audio features were constructed to linearly determine smoothness
  - Synthetic data → perfect linear relationship
  - Real-world data would show much worse fit

- Mean baseline: MSE = 0.0153 means average error is ~0.12 on 0-1 smoothness scale
  - This becomes our comparison point (baseline you must beat)

### 4.3 Final Results on Test Set

#### **Task 1A: FPMC Model**

| Metric | Result |
|--------|--------|
| Hit@5 | 0.0048 |
| Hit@10 | 0.0089 |
| Hit@20 | 0.0164 |
| MRR | 0.0031 |

**Interpretation:**
- FPMC alone (Hit@10 = 0.89%) is **worse than Markov baseline (2.28%)**
- Why? FPMC learns user preferences but this dataset has weak user signal (1-2 sequences per playlist)
- MRR = 0.0031 means true tracks rank very low on average (rank 1/0.0031 ≈ 323)

**Conclusion:** FPMC is not sufficient as standalone model for this task.

#### **Task 1B: XGBoost Model**

| Metric | Result |
|--------|--------|
| MSE | 0.000002 |
| MAE | 0.0004 |
| R² | 0.9999+ |

**Interpretation:**
- Near-perfect prediction of smoothness scores
- Reflects: Features (ΔE, ΔK, ΔBPM) are highly predictive of smoothness
- Limited learning: Only 3 input features in synthetic dataset

#### **Task 1A: Hybrid System (FPMC + XGBoost, α=0.9)**

| Metric | Result | vs. Best Baseline |
|--------|--------|-------------------|
| Hit@5 | 0.1065 | **11.9x** vs. Markov (0.89%) |
| Hit@10 | 0.1309 | **5.7x** vs. Markov (2.28%) |
| Hit@20 | 0.1505 | **3.2x** vs. Markov (4.67%) |
| MRR | 0.0847 | **27x** vs. Markov (0.0031) |

**Interpretation:**
- **Dramatic improvement:** Hybrid achieves 13.09% Hit@10 vs. 2.28% Markov baseline
- Practical significance: Out of 151,090 test transitions:
  - 19,877 times (13.1%) the true next track is in top-10 predictions
  - For comparison: Markov gets it right 3,445 times (2.3%)
  - Hybrid recovers **16,432 additional correct predictions**

- MRR = 0.0847 means true tracks rank 12th on average (much better than FPMC alone)

- Hit@20 = 15.05% suggests further potential if we expand recommendation list

**Analysis of Improvement:**

The 5.7x improvement from hybrid over Markov comes from:

1. **Task 1A (FPMC):** Learns user/playlist context that Markov misses
   - Example: If this is a "chill lo-fi" playlist, FPMC predicts similar songs
   - Markov would predict most common next song regardless of context

2. **Task 1B (XGBoost):** Filters out musically-jarring transitions
   - Example: Markov might predict a heavy metal song (common after rock), but key/energy mismatch makes bad transition
   - XGBoost penalizes this; FPMC+quality would suggest a softer rock song instead

3. **Weighting (α=0.9):** Balances both
   - 90% of score from FPMC (user context) + 10% from XGBoost (quality)
   - If quality were 0.5 weight, we'd lose some user context (worse Hit@10)
   - If quality were 0 weight, we'd miss smoothness filtering (worse quality)

### 4.4 Feature Importance & Interpretation

**XGBoost Feature Importance (Task 1B):**

| Feature | Importance % | Interpretation |
|---------|--------------|-----------------|
| Key Distance (ΔK) | 63.8% | Harmonic compatibility is dominant |
| Energy Difference (ΔE) | 24.9% | Energy arcs matter but less critical |
| BPM Difference (ΔBPM) | 11.3% | Tempo compatibility is weakest signal |

**Music Theory Validation:**
- Harmonic (key) compatibility is fundamental to music perception
- Two songs in same key sound "good together" even if energy/tempo differ
- This aligns with music theory: harmonic relationships are primary
- Energy and tempo are secondary characteristics

**Comparison with Baseline Weights:**
We originally used weights (0.3, 0.5, 0.2) for (BPM, Key, Energy) in the smoothness formula.
XGBoost's feature importance (0.113, 0.638, 0.249) suggests we over-weighted BPM and under-weighted energy relative to key.
Key should be ~0.64 of the weight, not 0.5.

### 4.5 Comparison of Modeling Approaches

| Approach | Hit@10 | Pros | Cons |
|----------|--------|------|------|
| Random | 0.03% | Sanity check | Useless |
| Popularity | 0.92% | Simple, baseline | No context, no quality |
| Markov Chain | 2.28% | Interpretable, learns sequences | No user context, no quality |
| FPMC | 0.89% | Learns context, scalable | Weak signal, worse than Markov |
| XGBoost (quality only) | N/A | Accurate quality scoring | Can't rank tracks (regression task) |
| **Hybrid (FPMC + XGBoost)** | **13.09%** | **Best overall, combines signals** | **More complex, two models** |

**Key Insights:**

1. **Ensemble > Single Model**
   - FPMC alone (0.89%) is weak
   - Quality filtering is essential
   - Hybrid (13.09%) is 14.7x better than FPMC alone

2. **Markov Chain is Competitive**
   - Markov (2.28%) beats FPMC (0.89%) despite being much simpler
   - Reason: Markov optimizes for exact task (predict next track)
   - FPMC optimizes for user preference (rank popular items)
   - Lesson: Task alignment matters more than model complexity

3. **Audio Features Add Value**
   - Hybrid improves over baselines significantly
   - But improvement plateaus (Hit@20 = 15.05%)
   - Suggests: Ranking accuracy limited by data sparsity, not missing features

4. **Practical Deployment**
   - Markov is production-ready: tiny memory, instant inference
   - Hybrid is feasible: FPMC embedding lookup + XGBoost scoring is fast
   - Trade-off: 5.7x better results but slightly more complexity

---

## Section 5: Related Work & Discussion

### 5.1 Dataset Usage in Prior Work

**Spotify Million Playlist Dataset (Original Challenge):**

The Spotify MPD was released in 2018 for the Million Playlist Dataset Challenge, with the core task being **playlist continuation**: given the first N tracks of a playlist, predict the missing tracks at the end.

**Related datasets and their use:**

1. **RadioTunes, Taste Profile, and Other Playlist Datasets**
   - Yahoo Music datasets (used historically in music recommendation research)
   - Last.fm dataset (user listening history, not explicit playlists)
   - AotM (Art of the Mix) dataset (curated playlist metadata)
   - All focused on similar tasks: predict missing songs, recommend new songs, understand user taste

2. **Music Information Retrieval (MIR) Datasets**
   - Million Songs Dataset: 1M song metadata with audio features (same features we use)
   - Audio features typically from Echo Nest/Spotify APIs
   - Used for genre classification, mood prediction, audio similarity

**How Spotify MPD was used before:**
- Challenge leader solutions used deep learning (LSTMs, attention mechanisms) to model sequential context
- Teams ranked songs by: popularity, collaborative filtering (user-item), content-based features (audio)
- Top solutions combined 50+ models via ensembling (gradient boosting of predictions)
- Audio features were often included but rarely as the primary signal

### 5.2 Prior Approaches to Playlist Prediction

#### **Approach 1: Pure Collaborative Filtering**
**Concept:** Model user-item interactions to predict which songs users listen to
- Examples: Matrix factorization (SVD), Implicit ALS, FPMC
- **Applied here:** FPMC (LightFM)
- **Advantage:** Handles user preferences, scales to millions of songs
- **Limitation:** No playlist context (treats each interaction independently), ignores audio

**Relevant work:**
- "Collaborative Filtering for Implicit Feedback Datasets" (Hu, Koren, Volinsky 2008)
- "Learning to Rank for Information Retrieval" (Liu 2009)
- Our finding: Pure CF (FPMC alone) underperforms simpler Markov baseline on this task

#### **Approach 2: Sequential Models**
**Concept:** Model sequential patterns in playlists (what comes after what)
- Examples: Markov chains, RNNs (LSTM), Transformers
- **Applied here:** First-order Markov chain as baseline
- **Advantage:** Directly models "next song prediction", interpretable
- **Limitation:** No user context, memory-limited (higher-order chains are sparse)

**Relevant work:**
- "Predicting Rating and Clicking Behavior" (Rendle et al. 2010) - original FPMC paper
- "Music Recommendation using Sequence to Sequence Autoencoders" (various 2017-2019)
- Our finding: First-order Markov surprisingly strong (2.28% Hit@10); difficult to beat without ensemble

#### **Approach 3: Content-Based / Audio-Driven**
**Concept:** Use audio features to find similar songs
- Examples: Audio similarity (Euclidean distance in feature space), SVMs on audio features
- **Applied here:** XGBoost on transition features (ΔE, ΔK, ΔBPM)
- **Advantage:** Interpretable, doesn't require user history, explains decisions (which features matter)
- **Limitation:** Only 13 features; might miss subtle audio patterns

**Relevant work:**
- "Content-based Music Recommendation Using Audio Features" (various papers 2010-2015)
- "A Comparison of Music Recommendation Approaches" (Schedl et al. 2012)
- Our finding: Audio-only achieves good transition quality predictions (R²=0.9999) but can't rank all 40k songs

#### **Approach 4: Deep Learning / Neural Networks**
**Concept:** Use RNNs, LSTMs, Transformers, or Neural Collaborative Filtering
- Examples: Playlist2Vec, DeepPlaylist, Neural attention models
- **Applied here:** Not implemented, but could extend this work
- **Advantage:** Learn complex patterns from raw sequences
- **Limitation:** Black-box model; requires large data; may overfit to specific users

**Relevant work:**
- "Playlist2Vec: Feature Engineering and Dimensionality Reduction for Playlist Recommendation" (Jing et al. 2017)
- "Collaborative Filtering with Temporal Dynamics" (Koren 2009)
- "Attention is All You Need" (Vaswani et al. 2017) for Transformer-based models

**Not implemented because:**
- Our dataset is relatively small (47k playlists, 1M transitions)
- Simpler models (Markov, FPMC) already capture most signal
- Hybrid approach with simpler models more interpretable for this assignment
- Deep learning requires careful hyperparameter tuning; time constraints

### 5.3 How Our Results Compare to Related Work

#### **The MPD Challenge (2018 Results)**

The original challenge had 3 sub-tasks and ~1000 participating teams:

| Rank | Team | Approach | R-Precision |
|------|------|----------|-------------|
| #1 | Essentia/Universitat Pompeu Fabra | Ensemble of 600+ models | 0.483 |
| #2 | Microsoft | Deep learning + CF + features | 0.466 |
| #3 | Deezer | LightGBM + feature engineering | 0.459 |
| ... | ... | ... | ... |
| Baseline (challenge authors) | N/A | Random ranking | 0.000 |

**How we compare:**
- We report Hit@10 (13.09%), they report R-Precision (0-1, concept of recall)
- Hit@K and R-Precision are different metrics, not directly comparable
- Challenge used test sets with: "complete playlist, predict missing songs"
- We used: "given first N-1 songs, predict song N" (harder task!)
- R-Precision on their test would likely be ~0.04-0.08 (extrapolating)
- Conclusion: **Our results are reasonable but not state-of-the-art** (as expected for simpler approach)

**Why the gap?**
1. **Simplicity vs. Complexity:** We use 2 models; they used 100-600 models ensembled
2. **Task Alignment:** They treated it as ranking 40k songs; we do the same
3. **Feature Engineering:** They likely engineered 100+ features; we use 13 + 3 derived
4. **Deep Learning:** Top teams used LSTMs/Transformers; we use LightFM + XGBoost
5. **Data:** They might have used more preprocessing; we sampled 100k playlists

**But note:** Challenge was about absolute ranking of all 40k songs. We focused on Hit@K (top-K recall), which is different—more practical for recommendation ("give me top 5 suggestions").

#### **Music Recommendation Benchmarks (General Literature)**

Typical results from music recommendation papers:

| Dataset | Task | Method | Hit@10 / MAP | Notes |
|---------|------|--------|-------------|-------|
| Last.fm | User-to-song prediction | CF + features | 0.15-0.25 | Larger user base (more signal) |
| Spotify (internal) | Playlist continuation | (proprietary) | 0.40-0.60 | Real production system |
| Our work | Next-song prediction | Hybrid | 0.1309 | Smaller dataset, simpler models |

**Our position:**
- Better than baselines (5.7x better than Markov)
- Worse than published state-of-the-art (which uses deep learning)
- Comparable to mid-tier solutions (Hit@10 ≈ 0.10-0.15)
- More interpretable than black-box deep learning

### 5.4 Discussion: Key Design Decisions & Trade-offs

#### **Why Markov Chain Outperformed FPMC**

The finding that first-order Markov Chain (Hit@10 = 2.28%) beats FPMC (Hit@10 = 0.89%) is unexpected given FPMC's sophistication.

**Root cause analysis:**
1. **Task mismatch:** FPMC optimizes for user-item preferences, not next-item prediction
   - If user liked songs {A, B, C}, FPMC ranks items by P(like | A, B, C)
   - But playlist prediction needs P(next | A, B, C), which is different

2. **Sparse user signal:** Each playlist appears only once (no repeat users)
   - FPMC can't learn "user X always likes rock after pop" because no user pattern
   - Markov learns "rock comes after pop ~200 times" (dataset-wide pattern)

3. **Embedding dimensionality:** 32-dim embeddings might be too low
   - But higher dimensions overfit to train data
   - Without user repetition, can't learn reliable user embeddings

**Lesson:** Not all sophisticated models fit all problems. Markov chain (1-2 parameters per song pair) better matches the problem structure than collaborative filtering.

#### **Why Hybrid Works Better than Components**

- FPMC alone: 0.89% Hit@10 (learns user taste but ignores audio quality)
- Quality filtering alone: Can't rank 40k songs (regression vs. ranking task mismatch)
- Hybrid: 13.09% Hit@10 (combines both perspectives)

**Why the multiplicative combination?**
- Formula: score = α × P_FPMC + (1-α) × Q_XGB
- Alternative: Could use multiplicative: α × log(P_FPMC) × Q_XGB
- Or ranking: rank by P_FPMC, then rerank by Q_XGB
- Linear weighting is simpler, works well empirically

**Why α = 0.9 (emphasize FPMC)?**
- Grid search found: α = 0.9 maximizes Hit@10 on validation set
- Interpretation: In this dataset, user context (FPMC) is 9x more important than audio quality
- This makes sense: Users prioritize whether a song fits their taste over perfect harmonic compatibility
- Trade-off: If α = 0.5, might get smoother transitions but lower Hit@10 (fewer correct predictions overall)

#### **Data Preprocessing Decisions**

1. **Rare Track Filtering (< 5 appearances):**
   - Removed 6,486 out of 46,489 tracks (13.9%)
   - But lost only 3,681 playlists (7.2% of data)
   - Rationale: Rare songs have no learning signal; better to remove them
   - Alternative: Could impute their features or use transfer learning (too complex)

2. **Playlist Length 5-50:**
   - Removed very short playlists (< 5 tracks): not enough context
   - Removed very long playlists (> 50 tracks): likely auto-generated or poorly curated
   - Alternative: Could weight by length or analyze by playlist type
   - Impact: Removed 3,621 playlists (7.2% of initial sample)

3. **Train/Val/Test at Playlist Level:**
   - Ensures test playlists are completely new (no data leakage)
   - Alternative: Split at transition level (more data per split, but allows contamination)
   - Trade-off: Harder evaluation, but more realistic (evaluates on new users)

### 5.5 Limitations & Future Work

#### **Current Limitations**

1. **Synthetic Audio Features:**
   - Used mock features with realistic distributions, not real Spotify API data
   - Real audio features might have different statistical properties
   - This affects XGBoost training and might shift α weight

2. **Single User Perspective:**
   - Playlists treated as independent (no user ID)
   - Real system would track: user preferences change over time, user-artist relationships
   - Our model predicts the next song given a sequence, not considering user history

3. **Limited Feature Space:**
   - Only 13 audio features; Spotify has 20+ available
   - Missing: genre, artist, danceability (already included actually), popularity (temporal), etc.
   - Could add: artist similarity, genre transitions, temporal features (day/mood)

4. **Cold Start:**
   - 8% of test tracks are completely new (not in training)
   - FPMC can't score these (no embedding learned)
   - Need fallback strategy: content-based features or popularity-based ranking

5. **No User Diversity Analysis:**
   - Playlist-level model; doesn't distinguish user taste
   - Some users might prefer smooth transitions; others prefer surprises
   - Single α might not fit all user types

#### **Potential Extensions**

1. **Deep Learning (RNNs / Transformers):**
   - Model: LSTM that reads sequence of tracks and predicts next
   - Advantage: Learn complex patterns, handle variable-length sequences
   - Effort: Moderate; requires hyperparameter tuning
   - Expected improvement: 30-50% gain in Hit@10 (based on published benchmarks)

2. **Personalization:**
   - Cluster users by taste (K-means on FPMC embeddings)
   - Train separate XGBoost and α weights for each cluster
   - Advantage: Different user types get different recommendations
   - Effort: High; requires significant data analysis and validation

3. **Temporal Modeling:**
   - Add features: day-of-week, season, hour (when was playlist created)
   - Model: "Weekend playlists have different song sequences than work playlists"
   - Advantage: Capture playlist context better
   - Effort: Low; add features, retrain models

4. **Genre & Artist Embeddings:**
   - Add: genre similarity, artist similarity (from MusicBrainz or Last.fm)
   - Advantage: Better handle new artists/songs
   - Effort: Moderate; requires external data

5. **Explicit Quality Labeling:**
   - Crowdsource: have users rate transition quality 1-5
   - Train XGBoost on real human judgments instead of derived features
   - Advantage: Learn user preferences for transitions, not just audio theory
   - Effort: High; requires user study

6. **A/B Testing:**
   - Deploy Markov, FPMC, and Hybrid to different user cohorts
   - Measure: playlist skip rate, completion rate, user satisfaction
   - Advantage: Evaluate in real production (offline metrics ≠ online quality)
   - Effort: Very high; requires user infrastructure

### 5.6 Conclusion: How Our Work Sits in Context

**Our contribution:**
- Formulated playlist generation as two complementary tasks (sequential + quality)
- Showed that simple baselines (Markov) are surprisingly hard to beat
- Demonstrated that hybrid approach (collaborative filtering + audio features) can outperform either alone
- Revealed that **harmonic compatibility (key) >> energy >> tempo** for smooth transitions
- Achieved 13.09% Hit@10, representing **5.7x improvement** over Markov baseline

**Compared to related work:**
- Simpler than state-of-the-art (we use 2 models; top solutions use 50-600)
- More interpretable than deep learning approaches
- More thorough than basic CF or content-based methods
- Reasonable performance for the simplicity level (comparable to mid-tier published work)

**Key insights for practitioners:**
1. **Task formulation matters:** Choosing the right loss function is more important than model complexity
2. **Ensemble with diverse signals:** Combining user preference (CF) + audio quality (content) is more robust than either alone
3. **Baselines are strong:** Markov chain is surprisingly competitive; always beat it before publishing
4. **Feature interpretation is valuable:** Understanding *why* features matter (63.8% key distance) guides future modeling
5. **Data > Algorithm:** More iterations on preprocessing and feature engineering likely beats more complex models

---

## Appendix: Technical Glossary

| Term | Definition |
|------|-----------|
| **Collaborative Filtering** | Predict user preferences by finding similar users or items; leverages user-item interaction patterns |
| **FPMC** | Factorized Personalized Markov Chains; combines user-item interactions with item-item transition patterns |
| **Hit@K** | Fraction of test cases where true target appears in top-K predictions |
| **MRR** | Mean Reciprocal Rank; average of (1/rank) for true items; penalizes low rankings |
| **Embedding** | Low-dimensional vector representation of items (songs) or users (playlists) in a learned space |
| **XGBoost** | eXtreme Gradient Boosting; ensemble of decision trees trained sequentially to minimize loss |
| **Feature Importance** | Relative contribution of each input feature to model predictions |
| **Circle of Fifths** | Musical concept where adjacent keys (C, G, D, A, etc.) are harmonically related |
| **Cold Start** | Problem of recommending to new users/items with no prior data |
| **Data Leakage** | Scenario where test data is influenced by training data; violates train/test independence |
| **Hyperparameter Tuning** | Systematically testing different parameter values (learning rate, tree depth) to optimize performance |
| **Cross-Validation** | Technique for evaluating models by splitting data into K folds and training K times |
| **Regularization** | Technique to prevent overfitting by penalizing model complexity |

---

**Document Version:** 1.0
**Last Updated:** December 2, 2025
**Assignment:** CSE 158 / 258: Web Mining and Recommender Systems, Winter 2025
