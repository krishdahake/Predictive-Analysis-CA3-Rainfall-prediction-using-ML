# Rainfall Prediction Using Machine Learning

**CA3 Project | Predictive Analytics | Symbiosis Institute of Technology, Nagpur**

**Author:** Krish Dahake  
**PRN:** 23070521076  
**Programme:** B.Tech Computer Science

---

## 1. Problem Statement

Predict whether it will rain the next day (`RainTomorrow` = Yes / No) from daily weather observations recorded at stations across Australia.

This is a **binary classification** problem. Rainy days are much rarer than dry days, so the project focuses on rain-class precision, recall and F1-score instead of accuracy alone. Forecasts like this support agriculture, water management and weather-related planning.

## 2. Dataset

| Item | Details |
|---|---|
| Name | weatherAUS (Rain in Australia) |
| File | `weatherAUS.csv` |
| Size | 145,460 rows, 23 columns |
| Period | About 10 years of daily observations |
| Source | Australian Bureau of Meteorology records (Kaggle: "Rain in Australia") |
| Target | `RainTomorrow` (Yes = 1, No = 0) |

Features include location, temperatures, rainfall, evaporation, sunshine, wind gust and wind direction/speed, humidity, pressure, cloud cover and `RainToday`.

## 3. Methodology (KDD Process)

1. **Business understanding:** define the binary rain / no-rain forecasting task.
2. **Data selection:** load `weatherAUS.csv`; target is `RainTomorrow`, `Date` preserves time order.
3. **Data understanding (EDA):** inspect missing values, class distribution (about 3.5 : 1 dry to rainy) and the relationship between `Humidity3pm` and rain.
4. **Cleaning and preprocessing:**
   - Remove rows with a missing target (3,267 rows, leaving 142,193 records).
   - Convert `Date` and sort chronologically.
   - Median imputation for numeric columns; most-frequent imputation for categorical columns.
5. **Transformation and feature engineering:**
   - Extract `Month` and `Year` from `Date`.
   - Encode the target (No = 0, Yes = 1).
   - One-hot encode categorical features; standardise numeric features for Logistic Regression.
   - All steps are built in scikit-learn `Pipeline` / `ColumnTransformer` and **fitted on training data only** to prevent data leakage.
6. **Data mining (modelling):** three models trained on the same training data.
7. **Evaluation and knowledge presentation:** accuracy, rain precision, rain recall, rain F1-score, classification report and confusion matrices, plus a location-based prediction demo.

### Train-test split

A **chronological split by unique dates**: the earliest 80% of dates are used for training and the remaining 20% for testing (test period starts 8 August 2015).

| Set | Records |
|---|---|
| Training | 109,302 |
| Testing | 32,891 (25,680 dry days, 7,211 rainy days) |

### Models

| Model | Description |
|---|---|
| Baseline (Dummy Classifier) | Always predicts the most frequent class (No rain) |
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=100`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1` |

## 4. Results

Evaluated on the chronological test set (rain = positive class):

| Model | Accuracy | Rain Precision | Rain Recall | Rain F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.795 | 0.523 | 0.736 | 0.612 |
| Random Forest | 0.847 | 0.767 | 0.437 | 0.557 |
| Baseline (Dummy) | 0.781 | 0.000 | 0.000 | 0.000 |

**Key findings**

- The baseline reaches 78.1% accuracy but detects none of the rainy days, so accuracy alone is misleading on this imbalanced data.
- **Logistic Regression** gives the best rain F1-score (0.612) and the highest rain recall (0.736).
- **Random Forest** gives the highest accuracy (0.847) and rain precision (0.767) but misses more than half of rainy days (recall 0.437).
- Both machine learning models clearly outperform the baseline on rain detection.

## 5. Prediction Demonstration

The final notebook cell asks for a location name, takes the latest observation available for that location in the dataset, builds an input with the same training features and predicts rain for the next day using the best model (by rain F1-score).

Example: for **Sydney** (latest observation 25 June 2017) the model predicts **no rain**. This is a demonstration on historical data, not a live weather forecast.

## 6. Repository Structure

```
.
├── Rainfall_Prediction_KDD_Simplified_Interpretation_Conclusion.ipynb   # Source code
├── weatherAUS.csv                                                       # Dataset
├── README.md                                                            # Project description
├── Report.docx                                                          # Project report (optional)
└── Presentation.pptx                                                    # Project presentation (optional)
```

## 7. How to Run

### Option A: Google Colab

1. Open the notebook in Google Colab.
2. Upload `weatherAUS.csv` (or mount Google Drive).
3. Update the dataset path in the data-loading cell, for example:
   ```python
   df = pd.read_csv('weatherAUS.csv')
   ```
4. Run all cells from top to bottom (`Runtime > Run all`).

### Option B: Local machine

```bash
git clone <your-repository-url>
cd <repository-folder>
pip install -r requirements.txt
jupyter notebook
```

Open the `.ipynb` file, make sure `weatherAUS.csv` is in the same folder, and run all cells.

### Requirements

Create a `requirements.txt` file with:

```
pandas
numpy
matplotlib
scikit-learn
jupyter
```

## 8. References

- Fayyad, U., Piatetsky-Shapiro, G. and Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine*, 17(3), 37-54.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
- Young, J. and Young, A. Rain in Australia (weatherAUS) dataset, Kaggle; source: Australian Bureau of Meteorology.
