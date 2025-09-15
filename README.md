![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://github.com/surpiya19/rollerCoaster/actions/workflows/main.yml/badge.svg)

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

### ✅ ML Model Exploration (`mlmodel.py`)
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
##### All outputs can be found in the ***plots*** folder.
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

| Model                 | MAE (log scale) | RMSE (log scale) | R² (log scale) | Notes                                                   |
| --------------------- | --------------- | ---------------- | -------------- | ------------------------------------------------------- |
| **Linear Regression** | High            | High             | Near 0         | Struggles, predictions cluster near mean                |
| **Random Forest**     | Lower than LR   | Lower than LR    | Still low      | Handles non-linearities better                          |
| **XGBoost**           | **1.79**        | **3.56**         | **-0.04**      | Best model so far but still underpredicts extreme costs |

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

```
rollerCoaster/
│
├── .devcontainer
│   └── devcontainer.json         # Dev container setup 
│
├── .github/workflows/
│   └── main.yml                  # CI/CD workflow
│
├── data/
│   ├── coaster_db.csv            # Raw dataset
│   └── coaster_db_clean.csv      # Cleaned dataset
│
├── notebooks/
│   ├── coaster_exploration.ipynb # EDA and visualization
│   └── mlmodel.ipynb             # Model experimentation
│
├── plots/                        # Generated plots
│   ├── correlation_heatmap.png
│   ├── cost_vs_speed.png
│   ├── height_vs_speed.png
│   ├── predictions.png
│   ├── speed_distribution.png
│   ├── xgboost_predictions.png
│   └── yearly_trend.png
│
├── scripts/
│   ├── coaster_analysis.py       # Main analysis pipeline
│   └── mlmodel.py                # ML model training script
│
│
├── subsets/                      # Filtered datasets
│   ├── Active_subset.csv
│   ├── Expensive_subset.csv
│   ├── Fast_subset.csv
│   ├── Modern_subset.csv
│   └── Tall_inversions_subset.csv
│
├── tests/
│   ├── test_coaster_analysis.py       # Tests for coaster_analysis.py
│   └── test_mlmodel.py                # Tests for mlmodel.py
│   └── test_coaster_plots.py          # Tests for plot generation
│
├── .flake8                       # Linting fix
├── .gitignore                    # Git ignore rules
├── Dockerfile                    # Docker file for containerized setup
├── Makefile                      # Local make tests  
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies



```
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

## 🧪 Tests:
All tests are written with **pytest** and live inside `tests/`.

| Test File                        | Function                | Purpose                                                                                                                             |
| -------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_coaster_analysis.py` | `test_clean_data()`    | Ensures data cleaning correctly converts numeric columns, drops duplicates, and standardizes dates                                  |
|                                  | `test_create_subsets()`| Verifies that subset CSVs (`Active`, `Modern`, `Fast`, `Tall_inversions`, `Expensive`) are created with the correct filtering logic |
|                                  | `test_generate_plots()`| Runs the plotting functions and checks that expected `.png` files are generated in `plots/`                                         |
| `tests/test_mlmodel.py`          | `test_train_and_evaluate()` | Runs ML training pipeline on a small sample dataset, ensures models train without errors, and that metrics are returned             |
|                                  | `test_generate_ml_plots()` | Ensures ML-specific plots (`predictions.png`, `xgboost_predictions.png`) are created in the output folder                           |



### Running tests with coverage:
- pytest -vv --cov=scripts --cov-report=term-missing
### To run pytest with the code:
- PYTHONPATH=$(pwd) pytest -v

## Docker Setup:

### Build and run in a container.
```
   - docker build -t rollercoaster .  
   - docker run -it --rm rollercoaster
```
---

## CI/CD:
- This project uses GitHub Actions (.github/workflows/main.yml) to:
   - Install dependencies
   - Run linting and tests on every push/pull request
   - Report coverage in CI logs
---

## 📈 Conclusion & Key Takeaways

- **Data Cleaning:** Successfully standardized coaster dataset, created meaningful subsets (Fast, Tall, Expensive, etc.)
- **Visualization:** Identified trends like the surge in coaster construction in the 1990s–2000s
- **Modeling:** Found that current features only weakly predict construction cost — more feature engineering (e.g., park size, country-level economic indicators) could improve performance
#### Potential Next Steps:
- Incorporate feature engineering
- Tune models using cross-validation & hyperparameter search
- Explore non-regression models (e.g., gradient boosting regressors with engineered features)

---