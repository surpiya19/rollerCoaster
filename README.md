![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://github.com/surpiya19/rollerCoaster/actions/workflows/main.yml/badge.svg)

# 🎢 Roller Coaster Cost Prediction
Roller coasters are some of the most ambitious engineering projects in the world, and their construction costs can range anywhere from a few million dollars to well over $100 million.

This project explores a simple question: **can we predict how much a roller coaster costs to build using its physical features and metadata?**

I started with exploratory data analysis for this data and saw how the factors like height, speed, inversions, year introduced, and manufacturer relate to construction cost, uncovering historical trends and industry differences. The work for this can be found here: ![EDA](https://github.com/surpiya19/rollerCoaster/blob/main/notebooks/coaster_exploration.ipynb)

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
│   ├── coaster_db_clean.csv      # Cleaned dataset -- without imputation
│   └── coaster_db_imputed.csv    # Cleaned dataset -- with imputations
│
├── notebooks/
│   └── coaster_exploration.ipynb     # EDA and visualization
│   
├── outputs/
│   ├── model_comparison.csv
│   └── model_metrics.csv    
│
├── plots/
│   ├── before_after_imputation_comparison.png
│   ├── correlation_heatmap.png
│   ├── cost_vs_speed.png
│   ├── model_comparison.png
│   ├── model_speed_distributions.png
│   ├── height_vs_speed.png
│   ├── predictions.png
│   ├── speed_distribution.png
│   └── yearly_trend.png
│
├── scripts/
│   ├── coaster_analysis.py       # Main analysis pipeline
│   ├── impute_costs.py           # Cost imputation pipeline
│   ├── modelling_report.py       # Metrics comparison before and after imputation
│   └── mlmodel.py                # ML models training script
│
├── refactoring/
│   ├── before.png                  # Code snippet example before refactoring
│   └── after.png                   # Code snippet example after refactoring
│   ├── changing_file_names.png     # Refactoring using pylance
│
├── subsets/                      # Filtered datasets
│   ├── Active.csv
│   ├── Expensive.csv
│   ├── Fast.csv
│   ├── Modern.csv
│   └── Tall_inversions.csv
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

## Data Pipeline
This repo provides a production-ready pipeline for cleaning, analyzing, imputing, and modeling roller coaster construction cost data.
The workflow is split across 4 main scripts, each serving a specific stage in the pipeline:

1. `coaster_analysis.py`: Data cleaning, filtering, exploratory analysis & visualizations.
2. `impute_costs.py`: Imputation of missing construction costs using a tuned Random Forest model.
3. `cost_prediction.py`: Model training (Linear Regression, Random Forest, XGBoost) on the imputed dataset, evaluation, and prediction plots.
4. `modeling_report.py`: Comparative study of model performance before vs after cost imputation, including feature importance analysis.

#### coaster_analysis.py:
- Cleans raw coaster_db.csv
- Saves coaster_db_clean.csv
- Generates summary stats & plots:
   - Speed distribution
   - Yearly trends of coaster introductions
   - Height vs speed scatter/regression
   - Cost vs speed scatter (log scale)
   - Correlation heatmap
- Creates filtered subsets (Active, Modern, Fast, etc.)

#### impute_costs.py
- Trains a Random Forest Regressor with GridSearchCV on available cost data.
- Uses engineered features (height², speed×height, inversions, park age, etc.).
- Imputes missing or invalid Cost values.
- Saves:
   - `coaster_db_imputed.csv` → dataset with Cost_clean and Cost_imputed flag
   - `models/rf_imputer.joblib` → fitted model

#### cost_prediction.py
- Trains and evaluates Linear Regression, Random Forest, and XGBoost on the imputed dataset.
- Saves:
   - outputs/model_metrics.csv → metrics (MAE, RMSE, R²)
   - plots/predictions.png → actual vs predicted plots

#### modeling_report.py
- Loads both raw (coaster_db_clean.csv) and imputed (coaster_db_imputed.csv) datasets.
- Trains the same 3 models on each.
- Compares performance:
   - `plots/before_after_imputation_comparison.png` → bar plot of metrics
- Extracts feature importance from Random Forest:
   - `plots/feature_importance.png`

---

## Visualization Outputs
##### All outputs can be found in the ***plots*** folder.
- `plots/speed_distribution.png` → Histogram of coaster speeds with mean/median markers.
- `plots/yearly_trend.png` → Line plot of coaster introductions over time.
- `plots/height_vs_speed.png` → Scatter showing correlation between height and speed.
- `plots/cost_vs_speed.png` → Scatter of cost vs speed (log scale).
- `plots/correlation_heatmap.png` → Correlation heatmap across numeric features.
-  `plots/feature_importance.png` → What affects the cost variable the most?

**For the mlmodel:**
- Correlation heatmap → `plots/correlation_heatmap.png`  
- Actual vs Predicted (all models) → `plots/predictions.png`  
- Detailed XGBoost evaluation → `plots/xgboost_predictions.png`  
- Before and After Imputation Metrics → `plots/before_after_imputation_comparison.png` 

---

## 🤖 Model Performance - Before and After Imputation
| Model                           | MAE (log scale) | RMSE (log scale) | R² (log scale)   | Notes                                                                  |
| ------------------------------- | --------------- | ---------------- | ---------------- | ---------------------------------------------------------------------- |
| **Linear Regression (Raw)**     | High (3.74)     | High (6.00)      | Negative (-0.37) | Struggles due to missing/inconsistent cost data                        |
| **Random Forest (Raw)**         | Moderate (3.27) | Moderate (5.90)  | Negative (-0.32) | Handles non-linearities better but still poor performance              |
| **XGBoost (Raw)**               | Moderate (3.23) | Moderate (6.23)  | Negative (-0.47) | Performs slightly better but underpredicts extreme costs               |
| **Linear Regression (Imputed)** | Low (0.67)      | Low (0.84)       | Positive (0.22)  | Benefits most from imputed data, predictions more stable               |
| **Random Forest (Imputed)**     | Low (0.66)      | Low (0.92)       | Low (0.07)       | Improved accuracy after imputation, handles non-linear features well   |
| **XGBoost (Imputed)**           | Moderate (0.92) | Moderate (1.36)  | Negative (-1.04) | Still struggles with some outliers, less stable than Linear Regression |

### 📉 Conclusion on model performance:
- On the raw dataset, all models perform poorly (negative R²), indicating that missing or inconsistent cost data severely affects predictive performance.
- After imputation, the models’ MAE and RMSE drastically decrease, showing more accurate cost predictions.
- Linear Regression shows the best overall improvement with a positive R² (0.22), suggesting it benefits most from the imputed data.
- Random Forest and XGBoost still have moderate performance on the imputed dataset, highlighting the complexity and variance in coaster cost data.
- Imputation significantly stabilizes predictions, demonstrating the importance of cleaning and filling missing cost values before modeling.

---

## ▶️ Usage

**Run the pipeline**:
```bash
python scripts/coaster_analysis.py
python scripts/impute_costs.py
python scripts/cost_prediction.py
python scripts/modeling_report.py
```

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

## 🐳 Docker Setup:

### Build and run in a container.
```
   - docker build -t rollercoaster .  
   - docker run -it --rm rollercoaster
```
---

## 👩🏽‍💻 CI/CD:
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