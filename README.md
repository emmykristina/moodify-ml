# Moodify – Music Emotion Classification

Moodify is a machine learning project exploring whether audio features can be used to classify music into four emotion categories: **Sad, Happy, Energetic, and Calm**.

The project compares unsupervised learning, traditional supervised machine learning, and deep learning. Several models were evaluated, with **XGBoost achieving the best test accuracy at 96.40%**.

The project also includes a small **Streamlit application** that recommends Spotify tracks based on a selected musical mood.

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

The app randomly selects five tracks from the chosen category and uses the **Spotify Web API** to retrieve artist and track information.

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

The datasets are not included in the repository.

Download the **278k Emotion Labeled Spotify Songs** dataset from Kaggle and place the CSV files inside:

```text
data/raw/
```

The project uses:

```text
278k_song_labelled.csv
278k_labelled_uri.csv
```

The first file is used for the machine learning analysis, while the URI version is used by the Streamlit application to connect recommendations to Spotify tracks.

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
streamlit run app.py
```

Select a musical mood and click **Recommend songs** to receive five Spotify recommendations.

---

## Project Structure

```text
moodify-ml/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── analysis.md
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
├── models/
│
└── src/
    └── moodify_ml/
```

Raw and processed datasets are excluded from Git.

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