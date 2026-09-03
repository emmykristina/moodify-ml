# Moodify – Music Emotion Classification

Moodify is a machine learning project exploring whether audio features can be used to classify music into four emotion categories: **Sad, Happy, Energetic, and Calm**.

The project compares unsupervised learning, traditional supervised machine learning, and deep learning. Several models were evaluated, with **XGBoost achieving the best test accuracy at 96.40%**.

The project also includes a small **Streamlit application** that recommends Spotify tracks based on a selected musical mood.

**🎧 Try it live:** [moodify-ml.streamlit.app](https://moodify-ml.streamlit.app)

---

## Results

| Model | Test Accuracy |
|---|---:|
| Dummy Classifier | 38.27% |
| Logistic Regression | 83.70% |
| ANN | 90.02% |
| Random Forest | 93.99% |
| **XGBoost** | **96.40%** |

The results show a clear improvement when moving from a simple baseline to non-linear models.

**XGBoost performed best overall**, while Calm was generally the easiest class to identify and Energetic the most difficult.

---

## Moodify App

A small Streamlit application was developed as an extension of the analysis.

The user selects one of four musical moods:

- **Melancholic**
- **Upbeat**
- **Energetic**
- **Calm**

These interface categories are mapped to the original dataset labels:

| App | Dataset label |
|---|---|
| Melancholic | Sad |
| Upbeat | Happy |
| Energetic | Energetic |
| Calm | Calm |

The names **Melancholic** and **Upbeat** are used in the application because they better describe musical characteristics without implying that a song itself is objectively sad or happy.

The application loads the trained XGBoost model and uses the audio features of each track to predict its emotion class. Tracks matching the user's selected mood are then used as candidates for the recommendations.

Spotify previews are displayed directly in the application, and starting a new preview automatically pauses the previous one.

Tracks with very high **speechiness** or a duration above **10 minutes** are filtered out to reduce the likelihood of recommending spoken-word content such as podcasts, audiobooks, or stand-up recordings.

---

## Getting Started

### 1. Clone the repository

```bash
git clone git@github.com:emmykristina/moodify-ml.git
cd moodify-ml
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

---

## Dataset Setup

The original dataset is the **278k Emotion Labeled Spotify Songs** dataset from Kaggle.

The project uses two dataset files:

```text
278k_song_labelled.csv
278k_labelled_uri.csv
```

`278k_song_labelled.csv` is used for the machine learning analysis and is not included in the repository.

`278k_labelled_uri.csv` contains the Spotify track identifiers and audio features required by the Streamlit application. This file is included in the repository so that the deployed application can generate model predictions and connect recommendations to Spotify.

---

## Spotify API Setup

The Streamlit application uses the Spotify Web API to retrieve artist and track metadata.

Create a Spotify Developer application and create a `.env` file in the project root containing:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

The `.env` file is excluded through `.gitignore` and should never be committed to the repository.

---

## Run the App

Start the Streamlit application with:

```bash
streamlit run app/app.py
```

Select a musical mood and click **Recommend songs** to receive five Spotify recommendations.

---

## Deployment

The app is deployed on [Streamlit Community Cloud](https://moodify-ml.streamlit.app), which rebuilds automatically on every push to `main`.

Deployment uses `app/requirements.txt` (a lean subset of the root
`requirements.txt`, without notebook-only dependencies like TensorFlow)
and reads the Spotify credentials from Streamlit's secrets manager
instead of a `.env` file. To deploy your own copy, add
`SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` under the app's
"Secrets" settings — see `.streamlit/secrets.toml.example` for the
format.

---

## Project Structure

```text
moodify-ml/
│
├── .streamlit/
│   └── secrets.toml.example
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── assets/
│   └── moodify_logo_transparent.png
│
├── data/
│   ├── raw/
│   │   └── 278k_labelled_uri.csv
│   └── processed/
│
├── docs/
│   └── analysis.md
│
├── models/
│   └── xgboost_model.json
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_machine_learning.ipynb
│   └── 05_deep_learning.ipynb
│
├── reports/
│   └── models/
│
├── README.md
└── requirements.txt
```

---

## Analysis

The project follows a complete machine learning workflow covering data understanding, data preparation, exploratory data analysis, unsupervised learning, supervised machine learning, and deep learning.

The analysis is divided into five notebooks:

1. `01_data_understanding.ipynb`
2. `02_data_preparation.ipynb`
3. `03_eda.ipynb`
4. `04_machine_learning.ipynb`
5. `05_deep_learning.ipynb`

For a summary of the methodology, experiments, and findings, see the [Analysis Documentation](docs/analysis.md).

---

## Limitations

The model uses audio features only and does not consider lyrics or lyrical meaning. Labels such as **Happy** and **Sad** should therefore be interpreted as musical characteristics associated with those emotions rather than a complete representation of a song's emotional meaning.

See the [Analysis Documentation](docs/analysis.md) for a more detailed discussion.

---

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- XGBoost
- TensorFlow / Keras
- Streamlit
- Spotify Web API
- Jupyter Notebook
- Git / GitHub