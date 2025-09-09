# 🎢 Roller Coaster Data Analysis
This project explores a dataset of rollercoasters from around the world, including details such as their height, speed, length, manufacturer, inversions, g-forces and opening dates. It also experiments with **predictive modeling** (Linear Regression, Random Forest, XGBoost) to estimate construction cost based on ride features.
This repository contains two complementary pipelines for exploring, cleaning, analyzing, and modeling roller coaster data.  

- **Data Exploration & Cleaning** → [`coaster_analysis.py`](./scripts/coaster_analysis.py)  
- **ML Cost Prediction** → [`mlmodel.py`](./scripts/mlmodel.py)  
- **Detailed Explanation** → Available in Jupyter Notebooks (`coaster_exploration.ipynb`,`mlmodel.ipynb`) for step-by-step insights and visualizations. 

---

## ⚙️ Features

### ✅ Data Pipeline (`coaster_analysis.py`)
1. **Data Cleaning**
   - Parses and standardizes dates.
   - Converts numeric columns (`Speed`, `Height`, `Cost`).
   - Cleans cost strings into consistent numerical values.
   - Drops duplicates.

2. **Filtering and Grouping**
   Creates meaningful subsets of the data:
   - **Active** → Coasters still operating.
   - **Modern** → Year introduced ≥ 2000.
   - **Fast** → Speed ≥ 70 mph.
   - **Tall_inversions** → Height ≥ 200 ft and at least 1 inversion.
   - **Expensive** → Cost ≥ $50M.

   Each subset is saved into the `subsets/` folder.

3. **Analysis & Visualizations**
   - Speed distribution histogram.
   - Yearly trend of coaster introductions.
   - Height vs. Speed scatter (with regression line).
   - Cost vs. Speed scatter (log scale).
   - Correlation heatmap.

   All plots are saved into the `plots/` folder.

4. **ML Model**
### Steps
a. **Data Loading**
   - Uses `coaster_db_clean.csv`
   - Keeps only `Cost`, `Gforce`, `Speed_mph`
   - Removes missing values

b. **Feature Engineering**
   - Target variable: `Cost` transformed via `log1p` for stability

c. **Models Trained**
   - Linear Regression (with scaling pipeline)
   - Random Forest Regressor
   - XGBoost Regressor

d. **Evaluation**
   - Metrics:  
     - MAE (Mean Absolute Error)  
     - RMSE (Root Mean Squared Error)  
     - R² (Coefficient of Determination)

---

## 📊 Example Outputs

- `plots/speed_distribution.png` → Histogram of coaster speeds with mean/median markers.
- `plots/yearly_trend.png` → Line plot of coaster introductions over time.
- `plots/height_vs_speed.png` → Scatter showing correlation between height and speed.
- `plots/cost_vs_speed.png` → Scatter of cost vs speed (log scale).
- `plots/correlation_heatmap.png` → Correlation heatmap across numeric features.

**For the mlmodel:**

- Correlation heatmap → `plots/correlation_heatmap.png`  
- Actual vs Predicted (all models) → `plots/predictions.png`  
- Detailed XGBoost evaluation → `plots/xgboost_predictions.png`  

---

## 🔍 Heatmap Analysis

**Correlation Matrix Highlights:**
- **Speed vs Gforce** → `0.46` (moderate positive correlation).
- **Cost vs Speed** → `-0.11` (very weak negative correlation).
- **Cost vs Gforce** → `-0.14` (very weak negative correlation).

👉 Implication: **Cost is not linearly related to ride features** → Linear models will struggle.

---

## 🤖 Model Comparison

### Linear Regression
- ✅ Simple, interpretable.  
- ❌ Predictions cluster near the mean → fails to capture variance.  

### Random Forest
- ✅ Handles non-linearities better, spreads predictions more.  
- ⚠️ Still underfits extremes.  

### XGBoost
- ✅ Best diagonal alignment among models.  
- ✅ Captures variance better.  
- ❌ Still underpredicts high costs.  
- ❌ **Overall performance is very poor**.

**XGBoost Metrics (log scale):**
- MAE = 1.79  
- RMSE = 3.56  
- R² = -0.04  

**XGBoost Metrics (actual cost scale):**
- MAE = 1.52B  
- RMSE = 6.27B  
- R² = -0.06  

### 📉 Conclusion on XGBoost
- Negative R² values → performs worse than predicting the mean.
- Huge MAE/RMSE → errors are unacceptably large.
- Scatter plots confirm predictions are scattered far from the diagonal.
- Model is **not fit for purpose** without major feature engineering or alternative modeling.

---

## 📂 Project Structure

rollerCoaster/
│── data/
│ └── coaster_db.csv # Raw dataset
│── scripts/
│ └── coaster_analysis.py # Main analysis pipeline
│── plots/ # Generated plots
│── subsets/ # Filtered datasets (CSV)
│── models/ # (optional) ML training scripts
│── README.md # Project documentation


---

## ▶️ Usage

1. **Run the pipeline**:
   ```bash
   python scripts/coaster_analysis.py
   python scripts/mlmodel.py
   ```
2. **Outputs**:

- Clean dataset → data/coaster_db_clean.csv
- Subsets → CSVs in subsets/
- Plots → PNGs in plots/

3. **🚀 Potential Next Steps**:

- Perform feature engineering (e.g., interaction terms, non-linear transformations).
- Explore regularization models (Ridge, Lasso).
- Consider domain-specific features (e.g., ride type, manufacturer grouping).
- Evaluate tree-based ensembles with hyperparameter tuning.
