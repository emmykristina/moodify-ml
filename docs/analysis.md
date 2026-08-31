# Moodify – Analysis Documentation

This document provides a summary of the data analysis, machine learning experiments, results, limitations, and conclusions of the Moodify project.

For the complete implementation, code, visualizations, and model evaluations, see the notebooks in the `notebooks/` directory.

---

## Dataset

The original dataset contains approximately **278,000 tracks** represented by numerical audio features and classified into four emotion categories:

| Label | Emotion |
|---:|---|
| 0 | Sad |
| 1 | Happy |
| 2 | Energetic |
| 3 | Calm |

The analyzed audio features are:

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

## 1. Data Understanding

The initial analysis focused on understanding:

- Dataset structure
- Data types
- Missing values
- Duplicate rows
- Class distribution
- Feature distributions
- Potential relationships between audio characteristics and emotion

Initial exploration showed meaningful differences in features such as energy and valence, while considerable overlap was also present between the emotion classes.

Calm appeared relatively distinct, while Sad was more dispersed across the explored feature space.

Based on the initial exploration, Sad was hypothesized to be one of the more difficult classes to predict. This hypothesis was later contradicted by the supervised model results, where Energetic consistently proved more difficult to classify.

---

## 2. Data Preparation

The main preparation steps were:

- Removing unnecessary index columns
- Removing 1,678 duplicate rows
- Removing the undocumented `spec_rate` feature
- Investigating extreme track durations
- Separating features and target labels
- Creating a stratified 80/20 train-test split
- Scaling features where required

Tracks longer than 10 minutes were investigated rather than automatically removed. These tracks appeared more frequently within the Sad and Calm classes, and there was not enough evidence to treat them as erroneous data.

They were therefore retained for the machine learning analysis.

The final modeling dataset contains approximately **276,000 tracks and 10 audio features**.

---

## 3. Exploratory Data Analysis

EDA showed meaningful differences between the four emotion categories, but also considerable overlap.

### Energy

Energetic tracks generally showed the highest energy values, while Calm tracks generally showed the lowest.

Happy tracks also tended to have relatively high energy, while Sad showed a wider distribution.

### Valence

Happy tracks generally showed higher valence values, while Calm showed lower values.

However, substantial overlap existed between the classes, showing that valence alone is not sufficient for distinguishing emotion.

### Feature Correlations

Several notable relationships were identified:

- Energy and loudness: approximately **+0.80**
- Energy and acousticness: approximately **-0.78**
- Danceability and valence: approximately **+0.52**

The strong relationships between some features indicate that different audio characteristics may capture related aspects of a track.

All documented features were retained because correlation alone was not considered sufficient reason to remove them.

### Statistical Testing

Kruskal-Wallis tests were performed for energy and valence.

Both tests showed statistically significant differences between the four emotion classes.

This supports the observation that these features contain information related to the emotion labels, although neither feature can separate all four classes independently.

---

## 4. Unsupervised Learning

### PCA

Principal Component Analysis was used to explore whether the audio features contained lower-dimensional structure related to emotion.

The first two principal components explained approximately:

- PC1: **34.6%**
- PC2: **12.8%**
- Combined: **47.4%**

Some structure related to emotion could be observed.

Calm appeared relatively distinct in parts of the PCA space, while Happy and Energetic showed considerable overlap. Sad occupied a more dispersed region.

However, the first two principal components were not sufficient to clearly separate all four classes.

### K-Means

K-Means clustering was performed using four clusters to investigate whether the audio features naturally formed groups corresponding to the four emotion labels.

The resulting clusters showed some relationship with the predefined labels.

For example, one cluster contained a relatively large proportion of Calm tracks, while another contained a high proportion of Happy tracks.

However, the clusters did not reproduce the four emotion categories clearly.

This suggests that the audio features contain meaningful structure, but the predefined emotion labels cannot be recovered through simple clustering alone.

---

## 5. Supervised Machine Learning

Four supervised approaches were evaluated, including a simple baseline.

| Model | Test Accuracy |
|---|---:|
| Dummy Classifier | 38.27% |
| Logistic Regression | 83.70% |
| Random Forest | 93.99% |
| XGBoost | **96.40%** |

### Dummy Classifier

The Dummy Classifier achieved:

**Accuracy: 38.27%**

It was used as a baseline to show how much the trained models improved over a simple prediction strategy based on the class distribution.

### Logistic Regression

Logistic Regression achieved:

**Accuracy: 83.70%**

This demonstrated that the audio features already contain substantial predictive information even when using a relatively simple linear model.

Calm was the easiest class for the model to identify.

Energetic was the most difficult and was frequently confused with Happy.

This contradicted the initial hypothesis that Sad would be the most difficult class.

### Random Forest

Random Forest achieved:

**Accuracy: 93.99%**

The substantial improvement over Logistic Regression suggests that non-linear relationships are important for distinguishing the emotion classes.

The most important features according to the Random Forest model included:

- Energy
- Instrumentalness
- Acousticness
- Danceability
- Loudness

Feature importance describes how much features were used by the model when making splits. It should not be interpreted as causality, and correlated features may share importance.

Energetic remained the most difficult class, although its classification improved substantially compared with Logistic Regression.

### XGBoost

XGBoost achieved:

**Accuracy: 96.40%**

This was the best result among the evaluated models.

XGBoost further reduced confusion between Energetic and Happy and produced strong performance across all four classes.

The proportion of Energetic tracks classified as Happy decreased across the models:

- Logistic Regression: approximately **21.55%**
- Random Forest: approximately **10.05%**
- XGBoost: approximately **5.76%**

XGBoost was therefore selected as the strongest supervised model in the project.

---

## 6. Deep Learning

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

The network contained **2,916 parameters**.

Early stopping was used to monitor validation loss and restore the best model weights.

The ANN achieved:

**Test Accuracy: 90.02%**

The model outperformed Logistic Regression but did not reach the performance of Random Forest or XGBoost.

As with the traditional machine learning models:

- Calm was the easiest class to identify.
- Energetic was the most difficult.
- Energetic was primarily confused with Happy.

The training and validation results did not show clear signs of overfitting.

Overall, the ANN successfully learned non-linear relationships between the audio features and emotion labels. However, the tree-based ensemble models were more effective for this structured tabular dataset.

This demonstrates that **deep learning does not necessarily outperform traditional machine learning methods**.

---

## Model Comparison

The final model comparison was:

| Model | Test Accuracy |
|---|---:|
| Dummy Classifier | 38.27% |
| Logistic Regression | 83.70% |
| ANN | 90.02% |
| Random Forest | 93.99% |
| **XGBoost** | **96.40%** |

Performance improved substantially when moving from the baseline and linear model toward non-linear approaches.

The ANN performed well but remained below both tree-based ensemble models.

XGBoost achieved the strongest overall performance.

---

## Key Findings

The main findings from the project are:

- Audio features contain substantial predictive information about the dataset's emotion labels.
- No individual feature is sufficient to clearly separate all four emotion classes.
- Unsupervised learning revealed meaningful structure, but did not reproduce the predefined emotion labels.
- Non-linear supervised models clearly outperformed the linear baseline.
- XGBoost achieved the highest test accuracy at **96.40%**.
- Calm was generally the easiest class to identify.
- Energetic was consistently the most difficult class.
- Energetic tracks were most commonly confused with Happy tracks.
- Tree-based ensemble models performed particularly well on this structured dataset.
- The ANN performed well but did not outperform Random Forest or XGBoost.

---

## Limitations

An important limitation concerns what the emotion labels actually represent.

The models use numerical audio characteristics but do not consider:

- Lyrics
- Lyrical meaning
- Musical context
- Individual listener interpretation

This is particularly important for the labels **Happy** and **Sad**.

For example, a track may contain low energy, low tempo, or other characteristics associated with the Sad category without actually having sad lyrics or being perceived as emotionally sad by a listener.

The labels should therefore primarily be interpreted as **musical characteristics associated with an emotion**, rather than a complete representation of the emotional meaning of a song.

This limitation is less pronounced for categories such as Energetic and Calm, which have a more direct relationship with measurable audio characteristics.

---

## Future Development

Possible future improvements include:

- Hyperparameter tuning
- More detailed model comparison
- Combining audio features with song lyrics
- Improving the definition of subjective emotion categories
- Exploring additional audio information
- Personalized recommendations
- Further development of the Streamlit application
- Using raw or time-segmented audio data
- Exploring temporal models such as TCNs

Combining **audio characteristics with lyrical information** would be particularly valuable for categories such as Happy and Sad, where audio features alone cannot fully represent the emotional meaning of a song.

Temporal models such as TCNs would require raw or time-segmented audio data rather than the aggregated tabular features used in the current project.

---

## Conclusion

Moodify demonstrates that numerical audio features can provide strong predictive information for the emotion labels used in the dataset.

The unsupervised analysis showed that the audio features contain meaningful structure, but the emotion categories could not be clearly recovered through PCA or K-Means alone.

Supervised learning produced substantially stronger results. Logistic Regression reached 83.70% accuracy, while Random Forest increased this to 93.99%. XGBoost achieved the strongest overall performance with **96.40% test accuracy**.

The ANN achieved 90.02%, demonstrating that deep learning can successfully learn non-linear relationships in the data, but also showing that deep learning does not automatically outperform traditional machine learning methods on structured tabular data.

Finally, the project highlights an important distinction between predicting a predefined emotion label from audio characteristics and understanding the full emotional meaning of music.

The results therefore demonstrate both the potential and the limitations of machine learning for music emotion classification.