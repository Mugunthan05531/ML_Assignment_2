# 🍷 Wine Quality Classification - Machine Learning Assignment 2

## 🚀 Live Demo
**Click here to try the app:** [https://mugunthan05531-ml-assignment-2.streamlit.app](https://mugunthan05531-ml-assignment-2.streamlit.app)

---

## a) Problem Statement
Predict whether a white wine is of **"good"** quality (quality score ≥ 7) or **"bad"** quality (quality score < 7) based on its chemical properties. This is a **binary classification problem** using 12 physicochemical features.

---

## b) Dataset Description
- **Source**: UCI Machine Learning Repository - Wine Quality Dataset
- **Dataset Name**: White Wine Quality
- **Instances**: 4,898 white wine samples
- **Features**: 11 physicochemical properties + 1 target variable

**Feature List:**
| # | Feature Name | Description |
|---|--------------|-------------|
| 1 | Fixed Acidity | Concentration of non-volatile acids |
| 2 | Volatile Acidity | Concentration of volatile acids |
| 3 | Citric Acid | Amount of citric acid (preservative) |
| 4 | Residual Sugar | Sugar left after fermentation |
| 5 | Chlorides | Salt concentration |
| 6 | Free Sulfur Dioxide | Free SO₂ in wine |
| 7 | Total Sulfur Dioxide | Total SO₂ (free + bound) |
| 8 | Density | Mass per unit volume |
| 9 | pH | Acidity/alkalinity level |
| 10 | Sulphates | Sulfur compounds (preservative) |
| 11 | Alcohol | Alcohol percentage |
| 12 | Quality (Target) | Quality score (3-9), converted to binary |

**Class Distribution:**
- **Good Wine** (Quality ≥ 7): ~25% of samples
- **Bad Wine** (Quality < 7): ~75% of samples

---

## c) GitHub Repository
**🔗 Repository Link:** [https://github.com/Mugunthan05531/ML_Assignment_2](https://github.com/Mugunthan05531/ML_Assignment_2)

---

## d) Models Used

### Comparison Table - Binary Classification Results

| ML Model Name | Accuracy | Precision | Recall | F1 Score | AUC | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6969 | 0.3924 | 0.7311 | 0.5107 | 0.7737 | 0.3514 |
| Decision Tree | 0.7663 | 0.4759 | 0.7925 | 0.5947 | 0.8046 | 0.4731 |
| KNN | 0.8316 | 0.6424 | 0.5000 | 0.5623 | 0.8517 | 0.4656 |
| Naive Bayes | 0.7204 | 0.4158 | 0.7217 | 0.5276 | 0.7490 | 0.3756 |
| **Random Forest** | **0.8929** | **0.8794** | **0.5849** | **0.7025** | **0.9295** | **0.6602** |

### Comparison Table - Multi-class Classification Results

| ML Model Name | Accuracy | Precision | Recall | F1 Score | AUC | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5276 | 0.2228 | 0.2050 | 0.1978 | NaN | 0.2386 |
| Decision Tree | 0.5316 | 0.3448 | 0.2872 | 0.3058 | NaN | 0.2823 |
| KNN | 0.5276 | 0.3176 | 0.2576 | 0.2708 | NaN | 0.2702 |
| Naive Bayes | 0.4724 | 0.3065 | 0.3032 | 0.2851 | NaN | 0.2669 |
| **Random Forest** | **0.6796** | **0.5328** | **0.3896** | **0.4298** | **NaN** | **0.5073** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy 69.7%, high recall (73.1%) but low precision (39.2%). The model identifies many wines as "good" but with many false positives. AUC of 0.7737 indicates moderate discriminative ability. |
| **Decision Tree** | Accuracy 76.6%, best recall among models (79.3%). Good F1 score (59.5%). AUC of 0.8046 shows good discriminative power. Slight overfitting is evident. |
| **KNN** | Accuracy 83.2%, good precision (64.2%) but lower recall (50.0%). Strong AUC of 0.8517 shows excellent discriminative ability. Benefits from feature scaling. |
| **Naive Bayes** | Lower accuracy (72.0%), moderate precision (41.6%) and recall (72.2%). AUC of 0.7490. Independence assumption doesn't hold for correlated wine features. |
| **Random Forest** | ✅ **Overall Winner** - Highest Accuracy (89.3%), AUC (92.9%), F1 Score (70.3%), and MCC (66.0%). Excellent precision (87.9%). Handles feature interactions effectively. |

---

### Overall Winner
🏆 **Random Forest** achieves the best performance across all evaluation metrics, making it the most suitable model for this dataset. Its ensemble nature effectively captures complex interactions between wine chemical properties and provides the most reliable predictions.

---

## 🚀 How to Run the Application

### Prerequisites
- Python 3.7 or higher

### 1. Clone the Repository
```bash
git clone https://github.com/Mugunthan05531/ML_Assignment_2.git
cd ML_Assignment_2