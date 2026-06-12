Task 12: Feature Engineering

Objective:-
Improve model performance by engineering new features from the California Housing Dataset.

Steps Performed:-
1. Data Loading & Cleaning
Loaded housing.csv (20,640 rows, 10 columns)
Filled 207 missing values in total_bedrooms with the column median

2. New Interaction Features
Feature                                       Formula                             Purpose
rooms_per_household                           total_rooms / households            Avg. rooms per home
bedrooms_per_room                             total_bedrooms / total_rooms        Layout ratio
population_per_household                      population / households             Crowding indicator
income_x_agemedian_income * housing_          median_age                          Combined wealth/age effect

4. Encoding ocean_proximity
->Label Encoding: mapped 5 categories (<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN) to integers 0–4
->One-Hot Encoding: created 5 binary columns, one per category

5. Log Transformation (Skew Correction)
Applied np.log1p() to right-skewed columns:
total_rooms, total_bedrooms, population, households, median_house_value, population_per_household
Skewness dropped significantly (e.g., total_rooms: 4.15 → -1.08, population: 4.94 → -1.04), bringing distributions closer to normal.

Output:-
->EnhancedDataset.csv — original 10 columns + 16 engineered/encoded columns (26 total)
->feature_engineering.py — full preprocessing code

Tools:-
Pandas, Scikit-learn (LabelEncoder, OneHotEncoder)
