# 🎢 Roller Coaster Data Analysis
This project explores a dataset of rollercoasters from around the world, including details such as their height, speed, length, manufacturer, inversions, g-forces and opening dates.

## Steps
1. **Import & Inspection**: Loaded the dataset using Polars, checked shape, head, summary stats, missing values, and duplicates.
2. **Filtering & Grouping**: 
   - Found top manufacturers by coaster count.
   - Identified average speeds by manufacturer.
   - Filtered coasters with speed > 60 mph.
3. **Machine Learning**:
   - Built a simple Linear Regression model to predict speed from height and inversions.
   - Found a positive relationship between coaster height and speed.
4. **Visualization**:
   - Created a scatterplot of height vs. speed, colored by type.

## Findings
- Taller coasters generally have higher speeds.
- Certain manufacturers (e.g., Intamin, B&M) dominate in both number and performance.
- Simple ML model confirms that height is a strong predictor of speed.

## Limitations
- Many missing values limit analysis.
- Linear regression was a simple choice; more advanced models could improve accuracy.
