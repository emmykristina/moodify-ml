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

# Getting Started

## 1. Clone the repository

```bash
git clone git@github.com:emmykristina/moodify-ml.git
cd moodify-ml
```

## 2. Create a virtual environment

### Windows

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

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

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

# Project Structure

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

# Dataset

The original dataset contains approximately **278,000 tracks** represented by numerical audio features.

The target contains four emotion classes:

| Label | Emotion |
|---:|---|
| 0 | Sad |
| 1 | Happy |
| 2 | Energetic |
| 3 | Calm |

The analyzed audio features include:

- Duration
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

The dataset contained no missing values.

After removing **1,678 duplicate rows**, approximately **276,000 tracks** remained.

The undocumented feature `spec_rate` was excluded because its meaning could not be reliably determined.

The classes are moderately imbalanced, with Happy as the largest class and Calm as the smallest.

---

# Analysis

The analysis is divided into five notebooks covering the complete workflow from initial data exploration to deep learning.

## 1. Data Understanding

The initial analysis focused on understanding:

- Dataset structure
- Data types
- Missing values
- Duplicate rows
- Class distribution
- Feature distributions
- Potential relationships between audio characteristics and emotion

Initial exploration suggested that features such as **energy and valence** differed noticeably between emotion classes, while considerable overlap was also present.

---

## 2. Data Preparation

The main preparation steps were:

- Removing unnecessary index columns
- Removing duplicate rows
- Removing the undocumented `spec_rate` feature
- Investigating extreme track durations
- Separating features and target labels
- Creating a stratified 80/20 train-test split
- Scaling features where required

The final modeling dataset contains approximately **276,000 tracks and 10 audio features**.

---

## 3. Exploratory Data Analysis

EDA showed meaningful differences between the emotion categories.

Some key observations:

- Energetic tracks generally have higher energy.
- Calm tracks generally have lower energy.
- Happy tracks tend to have higher valence.
- Sad tracks are more dispersed across the explored feature space.
- Energy and loudness have a strong positive correlation.
- Energy and acousticness have a strong negative correlation.
- Danceability and valence show a moderate positive relationship.

Statistical testing using the **Kruskal-Wallis test** showed significant differences between the four emotion classes for both energy and valence.

However, considerable overlap remained between the classes, showing that no individual audio feature was sufficient to explain emotion on its own.

---

# Machine Learning

## Unsupervised Learning

### PCA

Principal Component Analysis was used to explore whether the audio features contained lower-dimensional structure related to emotion.

The first two principal components explained approximately **47.4% of the total variance**.

Some structure could be observed, but the four emotion classes could not be clearly separated in two dimensions.

### K-Means

K-Means clustering with four clusters was also explored.

The clusters showed some relationship with the emotion labels, but did not reproduce the four predefined categories.

This suggests that the audio features contain meaningful structure, but the emotion labels are not naturally separated into four simple clusters.

---

## Supervised Learning

### Dummy Classifier

A Dummy Classifier was used as a baseline.

**Accuracy: 38.27%**

This reflects the performance of predicting primarily based on the most common class and provides a reference point for evaluating the trained models.

### Logistic Regression

Logistic Regression achieved:

**Accuracy: 83.70%**

This showed that the audio features already contain substantial predictive information even when using a relatively simple linear model.

### Random Forest

Random Forest achieved:

**Accuracy: 93.99%**

This was a substantial improvement over Logistic Regression and showed that non-linear relationships are important for distinguishing the emotion classes.

Feature importance indicated that features including **energy, instrumentalness, acousticness, danceability, and loudness** contributed strongly to the model's predictions.

Feature importance should not be interpreted as causality, and correlated features may share importance.

### XGBoost

XGBoost achieved the best result:

**Accuracy: 96.40%**

It also reduced one of the most common classification errors: confusion between **Energetic and Happy**.

XGBoost was therefore the strongest model evaluated in the project.

---

# Deep Learning

A feed-forward Artificial Neural Network was implemented using TensorFlow/Keras.

The architecture consisted of:

```text
10 input features
      ↓
Dense (64, ReLU)
      ↓
Dropout (0.2)
      ↓
Dense (32, ReLU)
      ↓
Dropout (0.2)
      ↓
Softmax (4 classes)
```

Early stopping was used to stop training when validation loss stopped improving.

The ANN achieved:

**Test Accuracy: 90.02%**

The model clearly learned useful non-linear relationships and outperformed Logistic Regression.

However, it did not outperform Random Forest or XGBoost.

This demonstrates that **deep learning does not necessarily outperform traditional machine learning methods**, particularly when working with structured tabular data.

---

# Key Findings

The main findings from the project are:

- Audio features contain substantial predictive information about the dataset's emotion labels.
- Non-linear models clearly outperform the linear baseline.
- XGBoost achieved the highest test accuracy at **96.40%**.
- Calm was generally the easiest class to identify.
- Energetic was consistently the most difficult class.
- Energetic tracks were most commonly confused with Happy tracks.
- Tree-based ensemble models performed particularly well on this structured dataset.
- The ANN performed well but did not outperform the strongest traditional machine learning models.

---

# Limitations

An important limitation of the project concerns what the emotion labels actually represent.

The models only use **audio characteristics** and do not consider lyrics, lyrical meaning, context, or the listener's subjective interpretation.

The labels **Happy** and **Sad** should therefore be interpreted with particular caution.

For example, a slow, low-energy track may contain audio characteristics associated with the Sad category even if the lyrics or overall emotional meaning of the song are not sad.

The categories should therefore primarily be understood as **musical characteristics associated with an emotion**, rather than a complete representation of the emotional meaning of a song.

This limitation is less pronounced for categories such as **Energetic** and **Calm**, which have a more direct relationship with measurable audio characteristics.

---

# Technologies

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

---

# Future Development

Possible improvements include:

- Hyperparameter tuning
- More detailed model comparison
- Combining audio features with song lyrics
- Improving the definition of subjective emotion categories
- Personalized recommendations
- Further development of the Streamlit application
- Using raw or time-segmented audio data
- Exploring temporal models such as TCNs

Combining **audio characteristics with lyrical information** would be particularly interesting for categories such as Happy and Sad, where audio features alone cannot fully represent the emotional meaning of a song.

---

# Conclusion

Moodify demonstrates that numerical audio features can be highly effective for classifying the emotion labels provided by the dataset.

Among the evaluated models, **XGBoost achieved the best overall performance with 96.40% test accuracy**.

At the same time, the project highlights an important distinction between predicting a predefined emotion label and actually understanding the emotional meaning of music.

The results therefore demonstrate both the potential and the limitations of using machine learning for music emotion classification.