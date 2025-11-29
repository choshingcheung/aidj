# AI DJ: Sequential Playlist Generation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

> **Intelligent playlist generation using sequential recommendation and audio compatibility modeling**

An end-to-end system that generates playlists with smooth track transitions by combining collaborative filtering (FPMC) with content-based audio feature analysis (XGBoost). Academic project for CSE 158/258 - Web Mining and Recommender Systems at UCSD.

---

## 🎯 Overview

This project tackles two complementary problems in music recommendation:

1. **Sequential Prediction**: What song should play next given playlist history?
2. **Transition Quality**: How well do two consecutive songs flow together?

By combining these approaches, we create playlists that are both personalized and musically coherent.

### Key Features

- 🎵 **FPMC-based sequential recommendation** for next-track prediction
- 🎼 **XGBoost regression** for transition quality assessment using 13 audio features
- 🔀 **Hybrid scoring system** combining collaborative + content-based signals
- 📊 **Comprehensive evaluation** against multiple baselines
- 🎧 **Audio demo generation** with intelligent crossfading (using pretrained Spleeter)

---

## 🚀 Quick Start

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Spotify Developer Account](https://developer.spotify.com/dashboard) (free)
- ~10GB disk space (for dataset)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aidj.git
cd aidj

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate aidj

# Verify installation
python src/utils/config.py
```

### Setup Spotify API

1. Create an app at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Copy your Client ID and Client Secret
3. Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Download Dataset

1. Visit [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
2. Download dataset (~5GB)
3. Extract to `data/raw/`

### Run the Notebook

```bash
jupyter notebook notebooks/ai_dj_main.ipynb
```

See [ROADMAP.md](ROADMAP.md) for detailed implementation checklist.

---

## 📊 Methodology

### Task Formulation

**Task 1A: Next Track Prediction**
- **Input**: Playlist history $(s_1, s_2, ..., s_t)$
- **Output**: Next song $s_{t+1}$
- **Evaluation**: Hit@K, AUC

**Task 1B: Transition Quality Regression**
- **Input**: Audio features of consecutive tracks $(s_i, s_j)$
- **Output**: Smoothness score $Q(s_i, s_j) \in [0, 1]$
- **Evaluation**: MSE, MAE, R²

### Models

#### 1. Factorized Personalized Markov Chains (FPMC)

Combines matrix factorization with first-order Markov chains:

$$\hat{y}_{u,i,j} = \langle V_u^U, V_i^I \rangle + \langle V_j^{LI}, V_i^{IL} \rangle$$

Where:
- First term: User-item preference (collaborative filtering)
- Second term: Sequential transition pattern (Markov chain)

Trained using Bayesian Personalized Ranking (BPR) loss.

**Reference**: Rendle et al. (2010) - *Factorizing Personalized Markov Chains for Next-Basket Recommendation*

#### 2. XGBoost Transition Quality Model

Gradient boosting regression on 13 audio feature differences:
- BPM (tempo) difference
- Key distance via circle of fifths
- Energy, valence, danceability deltas
- Harmonic compatibility, mode matching
- And 6 more acoustic features

**Reference**: McAuley et al. (2015) - *Image-based Recommendations on Styles and Substitutes* (adapted for audio)

#### 3. Hybrid System

Final scoring combines both models:

$$\text{score}(s_j | s_i) = \alpha \cdot P_{\text{FPMC}}(s_j | s_i) + \beta \cdot Q_{\text{XGB}}(s_i, s_j)$$

Where $\alpha + \beta = 1$ (tuned on validation set).

### Baselines

**Sequential Prediction:**
- Random selection
- Popularity ranking
- First-order Markov chain

**Transition Quality:**
- Mean predictor
- Linear regression

---

## 📁 Project Structure

```
aidj/
├── data/
│   ├── raw/              # Spotify Million Playlist Dataset
│   ├── processed/        # Sampled & cleaned playlists
│   ├── features/         # Extracted audio features from Spotify API
│   └── cache/            # API response cache
├── notebooks/
│   └── ai_dj_main.ipynb  # Main analysis notebook
├── src/
│   ├── models/           # Model implementations (FPMC, baselines)
│   ├── utils/            # Data loading, Spotify API, config
│   └── evaluation/       # Metrics and evaluation scripts
├── outputs/
│   ├── figures/          # Plots and visualizations
│   ├── audio/            # Generated playlist demos
│   └── results/          # Model performance results
├── environment.yml       # Conda environment specification
├── .env.example          # API credentials template
├── ROADMAP.md            # Implementation progress checklist
└── README.md
```

---

## 🎓 Academic Context

**Course**: CSE 158/258 - Web Mining and Recommender Systems
**Institution**: UC San Diego
**Dataset**: [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

### Deliverables

- ✅ Jupyter notebook (exported as HTML)
- ✅ 20-minute video presentation
- ✅ Peer grading report

---

## 📈 Expected Results

| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| Hit@10 | ~5% | ~15-25% | 3-5x |
| AUC | ~0.55 | ~0.75-0.85 | +0.20-0.30 |
| Transition R² | ~0.3 | ~0.6-0.7 | 2x |
| Playlist Smoothness | - | ~95% of human | - |

---

## 🛠️ Development

### Environment Management

```bash
# Activate environment
conda activate aidj

# Update environment from file
conda env update -f environment.yml --prune

# Export current environment
conda env export > environment.yml
```

### Running Tests

```bash
# Test configuration
python src/utils/config.py

# Test Spotify API
python src/utils/spotify_api.py

# Test data loading (requires dataset)
python src/utils/data_loader.py
```

---

## 📚 References

1. **Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L.** (2010). *Factorizing personalized markov chains for next-basket recommendation.* WWW 2010.

2. **McAuley, J., Targett, C., Shi, Q., & Van Den Hengel, A.** (2015). *Image-based recommendations on styles and substitutes.* SIGIR 2015.

3. **Hennequin, R., Khlif, A., Voituret, F., & Moussallam, M.** (2020). *Spleeter: a fast and efficient music source separation tool with pre-trained models.* ISMIR 2020. [*Note: Used only for audio demo*]

4. **Chen, S., Moore, J. L., Turnbull, D., & Joachims, T.** (2012). *Playlist prediction via metric embedding.* KDD 2012.

5. **Van den Oord, A., Dieleman, S., & Schrauwen, B.** (2013). *Deep content-based music recommendation.* NIPS 2013.

---

## 📝 License

This is an academic project for educational purposes. The code is provided as-is for reference.

**Dataset**: Spotify Million Playlist Dataset is subject to [AICrowd Challenge Terms](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

---

## 🙏 Acknowledgments

- **Spotify** for the Million Playlist Dataset
- **Deezer Research** for Spleeter pretrained models
- **Course Instructor and TAs** at UC San Diego
- **LightFM Contributors** for the recommendation library

---

## 📧 Contact

For questions about this project, please open an issue or reach out via the course forum.

---

**🎵 Built with passion for music and machine learning 🤖**
