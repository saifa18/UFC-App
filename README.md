# UFC Fighting Style Analysis Dashboard

**Thesis:** In the UFC, offensive striking edges are a stronger predictor of victory than grappling metrics.

---

## What This Project Is

An interactive data dashboard analyzing 7,756 UFC fights from 1994 to 2024. The project uses correlation analysis, feature ranking, and machine learning to answer one question: what actually separates winners from losers in the UFC?

The dashboard is built as both a standalone HTML file and a Streamlit web app, and tells a 7-visual story structured using the Martini glass narrative model from Kosara & Mackinlay (2013) and the sequence transition principles from Hullman et al. (2013).

---

## Project Files

```
ufc_app/
├── app.py                          ← Streamlit web app (main file)
├── requirements.txt                ← Python dependencies
├── merged_stats_n_scorecards.csv   ← Source dataset (7,756 fights)
├── fight_picture.png               ← Opening hero image
├── UFC-logo.png                    ← UFC logo (original)
├── UFC-logo-transparent.png        ← UFC logo (black bg removed, auto-generated)
├── ufc_dashboard_redesign_v2.html  ← Standalone HTML dashboard
└── build_ufc_dashboard_notebook_v2.ipynb  ← Jupyter notebook (source of truth)
```

---

## The 7 Visuals and What They Show

| # | Visual | Chart Type | Key Finding |
|---|--------|-----------|-------------|
| 1 | Body heatmap | Plotly silhouette | 63% of strikes land to the head |
| 2 | Winners vs Losers | Grouped bar | Winners land +11.7 more head strikes than losers |
| 3 | Style Evolution | Stacked area | Striking share grew from 37% → 65% (2002–2024) |
| 4 | Strike Differential | Two-line chart | Spearman ρ = 0.59, red wins 94% at +20 strikes |
| 5 | Control Time | Two-line chart | Spearman ρ = 0.38, relationship collapses in KO/TKO fights |
| 6 | Feature Ranking | Horizontal bar | Striking fills top 5, control time ranks 7th |
| 7 | Model Proof | Confusion matrix | Striking 84.7%, Grappling 74.1%, Combined 88.4% |

---

## Key Methodological Decisions

**Why Spearman correlation (not Pearson)?**
The relationship between striking differential and win rate is S-shaped, not linear — as Visual 4 shows clearly. Spearman measures monotonic relationships regardless of shape, making it the more honest choice.

**Why Logistic Regression (not Naive Bayes or KNN)?**
Logistic Regression performs best across all metrics, is the most interpretable, and avoids the independence assumption of Naive Bayes which is violated by our correlated fight stats.

**Why is the red corner baseline 65% (not 50%)?**
The UFC assigns red corner to the higher-ranked fighter by convention. Red wins 65% of fights purely because of this ranking system. All win-rate charts use 65% as the neutral baseline, not 50/50. The dataset is not balanced because this asymmetry is a real feature of the sport.

**Why is method_group excluded from the model?**
Knowing whether a fight ended by KO, submission, or decision is the same as knowing who won. Including it would be data leakage.

**Why is td_pct_diff excluded?**
57.6% missing values — too much to impute reliably.

**Why is ground_landed classified as Position-based (not Striking)?**
Ground strikes occur from a dominant grappling position. Classifying them as striking would artificially inflate the striking family's correlation scores.

---

## Dataset Notes

- **Source:** `merged_stats_n_scorecards.csv` — 7,756 fights, 61 columns
- **Date range:** 1994–2024
- **Key columns parsed:** All "X of Y" strike columns converted to numeric landed/attempted. Control time parsed from "MM:SS" format. All differential features computed as red minus blue.
- **Target variable:** `RedWin` — 1 if red corner won, 0 if blue corner won, NaN for draws/no-contests

---

## How to Run the Streamlit App

### Step 1 — Install dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```

### Step 2 — Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

### Step 3 — Deploy for free online

1. Create a free account at [github.com](https://github.com)
2. Create a new repository and upload all files in the `ufc_app/` folder
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Sign in with your GitHub account
5. Click **New app** → select your repository → set main file to `app.py` → click **Deploy**

Your app will be live at a public URL like `yourname-ufc-dashboard.streamlit.app` within 2–3 minutes.

---

## Storytelling Framework

This dashboard was designed following two academic papers on narrative visualization:

**Kosara & Mackinlay (2013) — "Storytelling: The Next Step for Visualization"**
- Martini glass structure: broad intuitive hook → narrow to statistical proof → open for interaction
- Opening visual (body silhouette) requires zero statistical knowledge — anyone can understand it immediately
- Each visual acts as a "step" in a clearly ordered sequence

**Hullman et al. (2013) — "A Deeper Understanding of Sequence in Narrative Visualization"**
- Every transition between visuals changes exactly one dimension (transition cost = 1)
- Visuals 4 and 5 form a deliberate parallelism pair — identical chart structure, different metric — which their research found specifically improves audience memory and comprehension
- Transition types used: Granularity (V1→V2), Temporal (V2→V3), Measure walk (V3→V4), Parallelism (V4↔V5), Specific-to-general (V5→V6), Causal (V6→V7)

---

## Model Results Summary

All models use Logistic Regression trained on 75% of fights, tested on 1,939 unseen fights.

| Feature Set | Correct | Accuracy | F1 | ROC-AUC |
|-------------|---------|----------|----|---------|
| Striking only | 1,642 / 1,939 | 84.7% | 0.886 | 0.920 |
| Grappling only | 1,437 / 1,939 | 74.1% | 0.821 | 0.785 |
| Combined | 1,715 / 1,939 | 88.4% | 0.913 | 0.944 |
| Naive baseline (always predict red) | — | 65.0% | — | — |

---

## Dependencies

```
streamlit >= 1.32.0
plotly >= 5.18.0
pandas >= 2.0.0
numpy >= 1.24.0
scipy >= 1.11.0
scikit-learn >= 1.3.0
Pillow >= 9.0.0   (for UFC logo transparency processing)
```
