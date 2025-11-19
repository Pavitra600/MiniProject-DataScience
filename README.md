## Traffic Accident Prediction – Data Preprocessing & Exploratory Data Analysis

This project involves preparing a traffic accident dataset for future predictive modeling. The notebook performs extensive data cleaning, feature engineering, and exploratory data analysis to identify relationships between different traffic-related features and accident occurrences.

## Objective

* Clean and organize the accident dataset
* Handle missing or duplicate records
* Fix inconsistent or unrealistic values
* Analyze influence of driver and road conditions on accidents
* Encode and scale features for model readiness
* Generate visual insights into patterns and correlations

## Dataset Information

* Original dataset: dataset_traffic_accident_prediction1.csv
* Cleaned dataset output: Cleaned_Accidents.csv
* Target feature: Accident
  (0 = No accident, 1 = Accident occurred)

## Features analyzed include:

Weather, Road Type, Road Condition, Time of Day, Vehicle Type, Speed Limit, Traffic Density, Driver Age and Experience, Road Light Condition
Number of Vehicles involved, Accident Severity.

## Tools and Libraries Used

Python, Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualizations
Label Encoding for categorical data transformation, StandardScaler for numerical scaling.

## Exploratory Data Analysis Performed

* Removed duplicate records
* Filled missing values using mode (categorical) and median (numerical)
* Corrected unrealistic values (example: speed limit capped at 150)
* Converted datatypes for model compatibility
* Reset index and saved cleaned dataset
* Generated descriptive statistics and dataset summary
* Created visualizations such as:
* Histograms for numerical features
* Count plots for categorical features
* Correlation heatmap
* Pairplots to observe relationships between key features
* Analysis of accident severity and accident counts

## Key Findings

* Higher traffic density and high speed values show stronger association with accidents
* Driver-related factors such as age and experience play an important role
* Accident rates vary based on lighting conditions and road type
* Correlation analysis reveals multiple interlinked features worth considering for prediction modeling


## Future Work

* Build and evaluate machine learning models to predict accident likelihood
* Perform hyperparameter tuning to improve model performance
* Conduct feature importance analysis for better insights
* Deploy the trained model as a user-friendly application

## Conclusion

The dataset was cleaned, analyzed, and transformed to understand key accident factors and prepare the data for machine learning.
The insights and visualizations highlight how traffic, driver behavior, and road conditions influence accident occurrence, providing a solid base for future prediction models.
