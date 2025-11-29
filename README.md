# AI DJ: Sequential Playlist Generation with Intelligent Track Transitions

**Course:** CSE 158/258 - Web Mining and Recommender Systems
**Assignment:** 2
**Dataset:** Spotify Million Playlist Dataset

## Project Overview

This project implements an intelligent DJ system that generates sequential playlists with smooth track transitions by combining:

1. **Sequential Recommendation (FPMC):** Predicts the next song given playlist history
2. **Transition Quality Assessment (XGBoost):** Learns musical compatibility between consecutive tracks
3. **Audio Generation (Spleeter):** Creates smooth crossfades based on learned transition quality

## Project Structure

```
aidj/
├── data/
│   ├── raw/              # Raw Spotify playlist JSON files
│   ├── processed/        # Cleaned and sampled playlists
│   ├── features/         # Extracted audio features
│   └── cache/            # Spotify API response cache
├── notebooks/
│   └── ai_dj_main.ipynb  # Main submission notebook
├── src/
│   ├── models/           # Model implementations
│   ├── utils/            # Helper functions
│   └── evaluation/       # Evaluation metrics
├── outputs/
│   ├── figures/          # Plots and visualizations
│   ├── audio/            # Generated playlist audio
│   └── results/          # Evaluation results
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aidj
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

1. Visit [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
2. Download the dataset (requires free AICrowd account)
3. Extract to `data/raw/`

### 5. Set Up Spotify API Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Copy your Client ID and Client Secret
4. Create a `.env` file in the project root:

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### 6. Run the Notebook

```bash
jupyter notebook notebooks/ai_dj_main.ipynb
```

## Predictive Tasks

### Task 1A: Next Track Prediction

**Input:** Playlist history $s_1, s_2, ..., s_t$
**Output:** Next song $s_{t+1}$
**Model:** Factorized Personalized Markov Chains (FPMC)
**Evaluation:** Hit@K, AUC

### Task 1B: Transition Quality Regression

**Input:** Audio features of consecutive tracks $(s_i, s_j)$
**Output:** Smoothness score $Q(s_i, s_j) \in [0, 1]$
**Model:** XGBoost Regression
**Evaluation:** MSE, MAE, R²

## Key Models

### 1. Factorized Personalized Markov Chains (FPMC)

Combines matrix factorization with Markov chains for sequential recommendation:

$$\hat{y}_{u,i,j} = \langle V_u^U, V_i^I \rangle + \langle V_j^{LI}, V_i^{IL} \rangle$$

- Captures both user preferences and sequential patterns
- Trained using Bayesian Personalized Ranking (BPR)

### 2. XGBoost Transition Quality Model

Learns compatibility function using 13 audio features:
- BPM difference
- Key distance (circle of fifths)
- Energy, valence, danceability differences
- Harmonic compatibility
- Mode matching
- And more...

### 3. Hybrid System

Combines FPMC + XGBoost for end-to-end playlist generation:

$$\text{score}(s_j | s_i) = \alpha \cdot P_{\text{seq}}(s_j | s_i) + \beta \cdot Q_{\text{trans}}(s_i, s_j)$$

### 4. Audio Mixing (Spleeter - Pretrained)

**Note:** We use Spleeter, a pretrained model by Deezer Research (Hennequin et al., 2020), for audio source separation. This is NOT our contribution—we only use it to demonstrate intelligent crossfading guided by our learned transition quality scores.

## Baselines

### Sequential Prediction:
- Random selection
- Popularity-based
- First-order Markov Chain

### Transition Quality:
- Mean baseline
- Linear regression

## Expected Results

- **Hit@10:** ~15-25% (vs. ~5% random baseline)
- **AUC:** ~0.75-0.85
- **Transition Quality R²:** ~0.6-0.7
- **Average Playlist Smoothness:** ~95% of human playlists

## Timeline

- **Week 1-2:** Data acquisition, preprocessing, EDA
- **Week 3-4:** Baseline models, FPMC implementation
- **Week 5-6:** XGBoost, hybrid system, audio demo
- **Week 7:** Presentation, polish, submission

## Key References

1. Rendle et al. (2010) - Factorizing Personalized Markov Chains
2. McAuley et al. (2015) - Image-based Recommendations on Styles
3. Hennequin et al. (2020) - Spleeter
4. Chen et al. (2012) - Playlist Prediction via Metric Embedding

## Deliverables

1. **Jupyter Notebook** (exported as HTML) - `workbook.html`
2. **20-minute Video Presentation** - Google Drive/YouTube link
3. **Peer Grading Report** (due 1 week later)

## License

This is an academic project for CSE 158/258 at UCSD.

## Acknowledgments

- Spotify for the Million Playlist Dataset
- Deezer Research for Spleeter pretrained models
- Course instructor and TAs for guidance
